# Project Context & Architecture Specification: Amazon Customer Support AI

## 1. Project Mission & Objective

The objective is to build an enterprise-grade Customer Support AI system modeled after Amazon's frontline support operations.

```
[Inbound Inquiry] ──> [Triage & Policy Retrieval] ──> [Resolution Generation] ──> [Escalation Check]
```

The system automates inbound customer inquiry triage, retrieves authoritative operational policies, generates empathetic and grounded customer resolutions, and determines whether an issue requires immediate human escalation.

## 2. Existing Workspace Assets

The repository contains two vetted, pre-processed datasets ready for use:

### 📄 `production_rag_knowledge_base.csv` (925 rows)
- **Purpose**: Source of truth for all customer service policies, procedural guidelines, and historical agent resolutions.
- **Schema**: `tweet_id` (identifier), `rag_context_text` (scrubbed, non-boilerplate operational response text).
- **State**: Customer handles are anonymized to `[CUSTOMER]`, tracking links are removed, and conversational boilerplate is stripped to leave only high-signal resolution guidance.

### 📄 `true_golden_eval_set.csv` (290 rows)
- **Purpose**: Stratified ground-truth benchmark for rigorous system evaluation.
- **Schema**: `tweet_id`, `text`, `gold_intent`, `gold_escalate`, `escalation_reason`.
- **Intent Distribution**:
  - Order Tracking & Delay (72 rows)
  - Account Access & General Inquiry (72 rows)
  - Refund & Return Request (72 rows)
  - Subscription & Billing (Prime) (61 rows)
  - Damaged or Missing Item (13 rows)
- **Escalation Distribution**: Exactly 22 rows are flagged (`gold_escalate = True`), capturing high-risk triggers such as fraud, severe anger, legal threats, or high-value compensation.

## 3. End-to-End System Lifecycle

The project executes sequentially across three pipeline scripts:

### 🛠️ Step 1: Vector Knowledge Base Indexing (`1_index_kb.py`)
- Ingests `production_rag_knowledge_base.csv`.
- Generates vector embeddings in configurable batches.
- Stores vectors and metadata (`tweet_id`) in a persistent local vector database (ChromaDB under `./chroma_db`).

### 🤖 Step 2: Retrieval-Augmented Generation Agent (`2_rag_agent.py`)
- Ingests raw customer queries.
- Retrieves the Top-K (default: K=3) most semantically relevant policy contexts from the vector store.
- Generates a context-grounded response adhering strictly to the @AmazonHelp support tone (empathetic, concise, solution-oriented).
- **Fallback Behavior**: If retrieved context lacks clear policy guidance, the agent routes the customer to direct support channels without hallucinating policies.

### ⚖️ Step 3: LLM-as-a-Judge Evaluation Suite (`3_evaluate_system.py`)
- Benchmarks the RAG pipeline against all 290 rows of `true_golden_eval_set.csv`.
- Employs an automated judge model to score four key metrics:
  - **Intent Alignment**: Verifies if the response accurately addresses the user's core intent.
  - **Escalation Precision & Recall**: Assesses if the model correctly flagged human hand-off triggers against `gold_escalate`.
  - **Faithfulness & Groundedness (1–5 scale)**: Evaluates the strict absence of hallucinated policies.
  - **Answer Relevance (1–5 scale)**: Measures the directness and utility of the provided solution.
- Exports complete summary analytics and a row-level breakdown to `final_evaluation_report.json`.

## 4. Engineering Constraints & Principles

- **Model Agnosticism**: Decouple all model endpoints, provider clients, and API keys through modular configuration files or environment variables. Do not hardcode specific model names or keys in the codebase.
- **Batching & Rate-Limiting Guardrails**: Any external embedding or inference call must support batching, exponential backoff logic, and configurable request pacing.
- **Idempotency & Checkpointing**: Ensure scripts can be safely re-run or resumed without corrupting vector stores or losing completed evaluation progress.
- **Strict Grounding**: The assistant must never invent return windows, refund amounts, or compensation rules that are not explicitly present in the retrieved context.
