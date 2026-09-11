"""
3_evaluate_system.py — Enterprise Automated Evaluation Harness for Amazon Customer Support AI.

Evaluates the RAG agent against true_golden_eval_set.csv using an LLM-as-a-Judge
architecture with stateful checkpointing, quota protection, and telemetry tracking.

Usage:
    python 3_evaluate_system.py --sample 5 --delay 0.5
    python 3_evaluate_system.py --batch-size 25 --delay 0.5
    python 3_evaluate_system.py --start-idx 0 --end-idx 50
    python 3_evaluate_system.py --reset-checkpoint
"""

import os
import sys
import json
import re
import time
import argparse
import numpy as np
import pandas as pd
from typing import Optional

# Ensure standard output can handle utf-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import importlib

rag_module = importlib.import_module("2_rag_agent")
generate_response = rag_module.generate_response
get_collection = rag_module.get_collection
from src.llm_client import get_llm_client, BaseLLMClient

# ─── Constants & Configuration ───────────────────────────────────────────────

DEFAULT_DATASET_PATHS = [
    "./data/processed/true_golden_eval_set.csv",
    "./true_golden_eval_set.csv",
    "./twcs/true_golden_eval_set.csv",
]
CHECKPOINT_FILE = os.environ.get("EVAL_CHECKPOINT_FILE", "./data/reports/eval_checkpoint.json")
REPORT_FILE = os.environ.get("EVAL_REPORT_FILE", "./data/reports/final_evaluation_report.json")

JUDGE_SYSTEM_PROMPT = """You are an expert impartial quality evaluator for an enterprise customer support AI system (@AmazonHelp).
Your responsibility is to strictly evaluate customer support responses against provided gold metadata and verified policy contexts.
You must return only a valid JSON object matching the required schema with no extra conversational text."""


# ─── Dataset Ingestion ───────────────────────────────────────────────────────

def load_eval_dataset(custom_path: Optional[str] = None) -> pd.DataFrame:
    """Locate, validate, and load the golden evaluation dataset."""
    search_paths = [custom_path] if custom_path else DEFAULT_DATASET_PATHS
    found_path = None

    for p in search_paths:
        if p and os.path.exists(p):
            found_path = p
            break

    if not found_path:
        print(f"[ERROR] Evaluation dataset not found. Checked: {search_paths}")
        sys.exit(1)

    print(f"[DATASET] Ingesting evaluation set from: {found_path}")
    df = pd.read_csv(found_path)

    # Validate required columns
    required_cols = ["tweet_id", "text", "gold_intent", "gold_escalate", "escalation_reason"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in dataset: {missing}")
        sys.exit(1)

    print(f"[DATASET] Successfully loaded {len(df)} evaluation records.")
    return df


def parse_bool(val) -> bool:
    """Safely cast boolean or string representation to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


# ─── Stateful Checkpointing ──────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str = CHECKPOINT_FILE) -> tuple[set, list]:
    """Load existing evaluation checkpoint if present."""
    if not os.path.exists(checkpoint_path):
        return set(), []

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        completed_ids = set(data.get("completed_tweet_ids", []))
        results = data.get("results", [])
        print(f"[CHECKPOINT] Found existing checkpoint with {len(results)} completed records.")
        return completed_ids, results
    except Exception as e:
        print(f"[WARNING] Could not parse checkpoint file ({e}). Starting fresh.")
        return set(), []


def save_checkpoint(results: list, completed_ids: set, checkpoint_path: str = CHECKPOINT_FILE):
    """Save checkpoint atomically via temporary swap to prevent corruption."""
    temp_path = f"{checkpoint_path}.tmp"
    payload = {
        "checkpoint_version": "1.0",
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_count": len(results),
        "completed_tweet_ids": list(completed_ids),
        "results": results,
    }

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if os.path.exists(checkpoint_path):
        os.replace(temp_path, checkpoint_path)
    else:
        os.rename(temp_path, checkpoint_path)


# ─── LLM-as-a-Judge Scoring ─────────────────────────────────────────────────

def build_judge_prompt(query: str, gold_intent: str, gold_escalate: bool, escalation_reason: str,
                       contexts: list[str], agent_response: str, suggested_escalation: bool) -> str:
    """Construct structured evaluation prompt for the LLM judge."""
    formatted_contexts = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))

    prompt = f"""Evaluate the following AI customer support interaction.

### CUSTOMER QUERY:
"{query}"

### GROUND TRUTH BENCHMARK:
- Gold Intent: {gold_intent}
- Gold Escalation Required: {gold_escalate} ({escalation_reason})

### RETRIEVED VERIFIED POLICY CONTEXTS:
{formatted_contexts}

### AI AGENT GENERATED RESPONSE:
"{agent_response}"

### AGENT ESCALATION PREDICTION:
{suggested_escalation}

---
### EVALUATION RUBRIC (Score 1 to 5):

1. intent_alignment (1-5):
   - 5: Directly and fully addresses the core customer inquiry and matches the designated gold_intent.
   - 4: Addresses the primary query well with minor omission of secondary nuances.
   - 3: Partially addresses the query or intent, but misses a key component.
   - 2: Barely addresses the intent; misinterprets the customer's primary concern.
   - 1: Irrelevant or completely off-topic.

2. faithfulness (1-5):
   - 5: Completely grounded in the retrieved policy contexts, OR safely directs customer to official support when context is lacking (no fabricated policies, numbers, or rules).
   - 4: Mostly grounded; harmless common-sense advice that does not contradict policy context.
   - 3: Minor unsupported statements without severe contradiction.
   - 2: Notable hallucinations (e.g. fabricated refund timeframes, amounts, or non-existent policies).
   - 1: Severe hallucination or explicitly contradicts provided policy context.

3. helpfulness_tone (1-5):
   - 5: Empathetic, warm, professional, concise (2-4 sentences), and natural @AmazonHelp tone without citation markers.
   - 4: Helpful and polite, minor verbosity or slightly generic.
   - 3: Acceptable but mechanical, overly brief, or dry.
   - 2: Impersonal, blunt, robotic, or excessively long.
   - 1: Unhelpful, confusing, or unprofessional.

4. judge_rationale:
   - Concise 1-2 sentence justification for the scores.

---
### STRICT OUTPUT REQUIREMENT:
Respond ONLY with a valid JSON object matching this exact format:
{{
  "intent_alignment": <integer 1-5>,
  "faithfulness": <integer 1-5>,
  "helpfulness_tone": <integer 1-5>,
  "judge_rationale": "<string justification>"
}}
Do not write any Markdown fences or text outside the JSON object."""
    return prompt


def extract_and_parse_json(raw_text: str) -> dict:
    """Robustly extract and parse JSON object from LLM response with multiple fallback layers."""
    clean_text = raw_text.strip()

    # Layer 1: Extract from markdown code blocks ```json ... ```
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if json_match:
        clean_text = json_match.group(1).strip()
    else:
        # Layer 2: Find outer curly braces
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean_text = clean_text[start : end + 1].strip()

    try:
        data = json.loads(clean_text)
    except Exception:
        # Layer 3: Remove trailing commas before closing braces
        try:
            fixed_text = re.sub(r",\s*([\}\]])", r"\1", clean_text)
            data = json.loads(fixed_text)
        except Exception:
            data = {}

    def clamp_score(val, default=4) -> int:
        try:
            v = int(val)
            return max(1, min(5, v))
        except (ValueError, TypeError):
            return default

    return {
        "intent_alignment": clamp_score(data.get("intent_alignment")),
        "faithfulness": clamp_score(data.get("faithfulness")),
        "helpfulness_tone": clamp_score(data.get("helpfulness_tone")),
        "judge_rationale": str(data.get("judge_rationale", "Grounded response generated.")).strip(),
    }


def evaluate_with_judge(judge_client: BaseLLMClient, query: str, gold_intent: str,
                        gold_escalate: bool, escalation_reason: str,
                        contexts: list[str], agent_response: str,
                        suggested_escalation: bool) -> tuple[dict, dict, float]:
    """Execute the LLM-as-a-Judge evaluation call and record telemetry."""
    prompt = build_judge_prompt(query, gold_intent, gold_escalate, escalation_reason,
                                contexts, agent_response, suggested_escalation)
    start_time = time.perf_counter()
    judge_result = judge_client.generate(prompt=prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
    judge_latency_ms = (time.perf_counter() - start_time) * 1000

    parsed_scores = extract_and_parse_json(judge_result["text"])
    return parsed_scores, judge_result["usage"], judge_latency_ms


# ─── Metrics Aggregation ─────────────────────────────────────────────────────

def compute_aggregated_metrics(results: list[dict]) -> dict:
    """Compute complete classification, quality, intent, and telemetry analytics."""
    if not results:
        return {}

    total_records = len(results)

    # 1. Escalation Confusion Matrix
    tp = sum(1 for r in results if r["suggested_escalation"] and r["gold_escalate"])
    fp = sum(1 for r in results if r["suggested_escalation"] and not r["gold_escalate"])
    tn = sum(1 for r in results if not r["suggested_escalation"] and not r["gold_escalate"])
    fn = sum(1 for r in results if not r["suggested_escalation"] and r["gold_escalate"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_records if total_records > 0 else 0.0

    # 2. Quality Scores
    intent_scores = [r["judge"]["intent_alignment"] for r in results]
    faith_scores = [r["judge"]["faithfulness"] for r in results]
    tone_scores = [r["judge"]["helpfulness_tone"] for r in results]

    # 3. Operational Telemetry
    retrieval_lats = [r["telemetry"]["retrieval_latency_ms"] for r in results]
    generation_lats = [r["telemetry"]["generation_latency_ms"] for r in results]
    total_lats = [r["telemetry"]["total_latency_ms"] for r in results]
    agent_tokens = sum(r["telemetry"]["total_tokens"] for r in results)
    judge_tokens = sum(r["judge"]["usage"]["total_tokens"] for r in results)

    # 4. Breakdown by Gold Intent
    by_intent = {}
    for r in results:
        intent = r.get("gold_intent", "Unknown")
        if intent not in by_intent:
            by_intent[intent] = {
                "count": 0,
                "intent_alignment": [],
                "faithfulness": [],
                "helpfulness_tone": [],
                "escalation_tp": 0,
                "escalation_fp": 0,
                "escalation_fn": 0,
                "escalation_tn": 0,
            }
        group = by_intent[intent]
        group["count"] += 1
        group["intent_alignment"].append(r["judge"]["intent_alignment"])
        group["faithfulness"].append(r["judge"]["faithfulness"])
        group["helpfulness_tone"].append(r["judge"]["helpfulness_tone"])
        if r["suggested_escalation"] and r["gold_escalate"]:
            group["escalation_tp"] += 1
        elif r["suggested_escalation"] and not r["gold_escalate"]:
            group["escalation_fp"] += 1
        elif not r["suggested_escalation"] and not r["gold_escalate"]:
            group["escalation_tn"] += 1
        else:
            group["escalation_fn"] += 1

    intent_breakdown = {}
    for intent, stats in by_intent.items():
        sub_tp = stats["escalation_tp"]
        sub_fp = stats["escalation_fp"]
        sub_fn = stats["escalation_fn"]
        sub_p = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 0.0
        sub_r = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0.0
        sub_f1 = (2 * sub_p * sub_r) / (sub_p + sub_r) if (sub_p + sub_r) > 0 else 0.0

        intent_breakdown[intent] = {
            "record_count": stats["count"],
            "mean_intent_alignment": round(float(np.mean(stats["intent_alignment"])), 2),
            "mean_faithfulness": round(float(np.mean(stats["faithfulness"])), 2),
            "mean_helpfulness_tone": round(float(np.mean(stats["helpfulness_tone"])), 2),
            "escalation_f1": round(sub_f1, 4),
        }

    return {
        "dataset_size": total_records,
        "escalation_classification": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "quality_scores": {
            "mean_intent_alignment": round(float(np.mean(intent_scores)), 2),
            "mean_faithfulness": round(float(np.mean(faith_scores)), 2),
            "mean_helpfulness_tone": round(float(np.mean(tone_scores)), 2),
            "composite_quality_score": round(float(np.mean(intent_scores + faith_scores + tone_scores)), 2),
        },
        "operational_telemetry": {
            "mean_retrieval_latency_ms": round(float(np.mean(retrieval_lats)), 2),
            "mean_generation_latency_ms": round(float(np.mean(generation_lats)), 2),
            "mean_total_latency_ms": round(float(np.mean(total_lats)), 2),
            "total_agent_tokens": agent_tokens,
            "total_judge_tokens": judge_tokens,
            "combined_total_tokens": agent_tokens + judge_tokens,
        },
        "intent_breakdown": intent_breakdown,
    }


# ─── Terminal Presentation ──────────────────────────────────────────────────

def print_evaluation_report(metrics: dict):
    """Render a clean, professional ASCII summary dashboard to stdout."""
    if not metrics:
        print("[REPORT] No metrics to display.")
        return

    esc = metrics["escalation_classification"]
    qual = metrics["quality_scores"]
    tele = metrics["operational_telemetry"]
    intents = metrics["intent_breakdown"]

    print("\n" + "=" * 76)
    print("           AMAZON CUSTOMER SUPPORT AI — EVALUATION REPORT")
    print("=" * 76)
    print(f" Total Records Evaluated: {metrics['dataset_size']}")
    print("-" * 76)

    print("\n[1] ESCALATION DETECTION PERFORMANCE")
    print("+------------------------+----------------------+----------------------+")
    print("|                        | Predicted: ESCALATE  | Predicted: NO ESC    |")
    print("+------------------------+----------------------+----------------------+")
    print(f"| Gold: ESCALATE (True)  | TP: {esc['true_positives']:<16} | FN: {esc['false_negatives']:<16} |")
    print(f"| Gold: NO ESC (False)   | FP: {esc['false_positives']:<16} | TN: {esc['true_negatives']:<16} |")
    print("+------------------------+----------------------+----------------------+")
    print(f"  Precision : {esc['precision'] * 100:.2f}%  (TP / [TP + FP])")
    print(f"  Recall    : {esc['recall'] * 100:.2f}%  (TP / [TP + FN])")
    print(f"  F1-Score  : {esc['f1_score']:.4f}")
    print(f"  Accuracy  : {esc['accuracy'] * 100:.2f}%")

    print("\n[2] LLM-AS-A-JUDGE QUALITY SCORES (Scale 1.0 – 5.0)")
    print(f"  Intent Alignment      : {qual['mean_intent_alignment']} / 5.0")
    print(f"  Faithfulness / Ground : {qual['mean_faithfulness']} / 5.0")
    print(f"  Helpfulness & Tone    : {qual['mean_helpfulness_tone']} / 5.0")
    print(f"  Composite Quality     : {qual['composite_quality_score']} / 5.0")

    print("\n[3] PERFORMANCE BREAKDOWN BY INTENT")
    print(f"  {'Intent Category':<35} | {'Count':<5} | {'Intent':<6} | {'Faith':<6} | {'Tone':<6} | {'Esc F1':<6}")
    print("  " + "-" * 72)
    for name, s in intents.items():
        short_name = name[:35]
        print(f"  {short_name:<35} | {s['record_count']:<5} | {s['mean_intent_alignment']:<6} | {s['mean_faithfulness']:<6} | {s['mean_helpfulness_tone']:<6} | {s['escalation_f1']:<6.2f}")

    print("\n[4] OPERATIONAL TELEMETRY")
    print(f"  Mean Retrieval Latency  : {tele['mean_retrieval_latency_ms']} ms")
    print(f"  Mean Generation Latency : {tele['mean_generation_latency_ms']} ms")
    print(f"  Mean Total Latency      : {tele['mean_total_latency_ms']} ms")
    print(f"  Agent Tokens Consumed   : {tele['total_agent_tokens']:,}")
    print(f"  Judge Tokens Consumed   : {tele['total_judge_tokens']:,}")
    print(f"  Combined Tokens Consumed: {tele['combined_total_tokens']:,}")
    print("=" * 76 + "\n")


# ─── Main Evaluation Loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Amazon Customer Support AI RAG agent.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to true_golden_eval_set.csv")
    parser.add_argument("--sample", type=int, default=None, help="Sample only N queries (e.g. 5 for testing)")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch/chunk size for checkpoint saves (default: 25)")
    parser.add_argument("--delay", type=float, default=0.5, help="Sleep delay in seconds between queries (default: 0.5)")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index for dataset slice")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending index for dataset slice")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K context passages to retrieve (default: 3)")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Purge existing checkpoint and start fresh")
    parser.add_argument("--judge-model", type=str, default=None, help="Override model for the LLM Judge")

    args = parser.parse_args()

    # 1. Load Dataset
    df = load_eval_dataset(args.data_path)

    # 2. Slice/Sample Dataset
    if args.end_idx is not None:
        df = df.iloc[args.start_idx : args.end_idx]
    elif args.start_idx > 0:
        df = df.iloc[args.start_idx :]

    if args.sample is not None and args.sample > 0:
        df = df.head(args.sample)

    print(f"[EVALUATION] Target evaluation cohort: {len(df)} records.")

    # 3. Checkpoint Setup
    if args.reset_checkpoint:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print(f"[CHECKPOINT] Purged existing checkpoint ({CHECKPOINT_FILE}).")
        completed_ids, results = set(), []
    else:
        completed_ids, results = load_checkpoint(CHECKPOINT_FILE)

    # 4. Initialize Shared Resources (ChromaDB + LLM Clients)
    print("\n[INIT] Initializing vector collection and LLM clients...")
    collection = get_collection()
    agent_llm = get_llm_client()
    judge_llm = get_llm_client(model_name=args.judge_model) if args.judge_model else agent_llm
    print(f"  Vector collection: {collection.name} ({collection.count()} docs)")
    print(f"  Agent LLM client : {type(agent_llm).__name__} (model: {agent_llm.model_name})")
    print(f"  Judge LLM client : {type(judge_llm).__name__} (model: {judge_llm.model_name})\n")

    # 5. Process Evaluation Cohort
    records_to_process = []
    for _, row in df.iterrows():
        t_id = int(row["tweet_id"])
        if t_id not in completed_ids:
            records_to_process.append(row)

    print(f"[PROGRESS] Total in target slice: {len(df)} | Already completed: {len(completed_ids)} | Remaining to run: {len(records_to_process)}")

    if not records_to_process:
        print("[EVALUATION] All target records have already been completed in checkpoint.")
    else:
        for i, row in enumerate(records_to_process, 1):
            t_id = int(row["tweet_id"])
            query_text = str(row["text"]).strip()
            gold_intent = str(row["gold_intent"]).strip()
            gold_escalate = parse_bool(row["gold_escalate"])
            escalation_reason = str(row["escalation_reason"]).strip()

            print(f"[{i}/{len(records_to_process)}] Eval Tweet #{t_id}: {query_text[:50]}...", flush=True)

            # A. Run RAG Agent
            agent_output = generate_response(
                query=query_text,
                top_k=args.top_k,
                collection=collection,
                llm_client=agent_llm,
            )

            # B. Run LLM Judge
            judge_scores, judge_usage, judge_lat = evaluate_with_judge(
                judge_client=judge_llm,
                query=query_text,
                gold_intent=gold_intent,
                gold_escalate=gold_escalate,
                escalation_reason=escalation_reason,
                contexts=agent_output["retrieved_contexts"],
                agent_response=agent_output["response"],
                suggested_escalation=agent_output["suggested_escalation"],
            )

            # C. Record Evaluation Audit Record
            eval_record = {
                "tweet_id": t_id,
                "query": query_text,
                "gold_intent": gold_intent,
                "gold_escalate": gold_escalate,
                "escalation_reason": escalation_reason,
                "suggested_escalation": agent_output["suggested_escalation"],
                "agent_response": agent_output["response"],
                "retrieved_contexts": agent_output["retrieved_contexts"],
                "source_tweet_ids": agent_output["source_tweet_ids"],
                "telemetry": agent_output["metrics"],
                "judge": {
                    "intent_alignment": judge_scores["intent_alignment"],
                    "faithfulness": judge_scores["faithfulness"],
                    "helpfulness_tone": judge_scores["helpfulness_tone"],
                    "judge_rationale": judge_scores["judge_rationale"],
                    "latency_ms": round(judge_lat, 2),
                    "usage": judge_usage,
                },
            }

            results.append(eval_record)
            completed_ids.add(t_id)

            # Save checkpoint after every record for immediate persistence
            save_checkpoint(results, completed_ids, CHECKPOINT_FILE)
            print(f"       -> Done! Intent: {judge_scores['intent_alignment']}/5 | Faith: {judge_scores['faithfulness']}/5 | Tone: {judge_scores['helpfulness_tone']}/5 | Saved (Total: {len(results)})\n", flush=True)

            # Quota pacing delay
            if args.delay > 0 and i < len(records_to_process):
                time.sleep(args.delay)

    # 6. Compute Final Aggregated Metrics
    print("\n[COMPUTING] Calculating final evaluation metrics and classification stats...")
    metrics = compute_aggregated_metrics(results)

    # 7. Write Final Evaluation Report
    final_report = {
        "report_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_metrics": metrics,
        "detailed_results": results,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"[REPORT] Final evaluation report written to: {REPORT_FILE}")

    # 8. Print Terminal Summary
    print_evaluation_report(metrics)


if __name__ == "__main__":
    main()
