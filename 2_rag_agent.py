"""
2_rag_agent.py — Retrieval-Augmented Generation Agent for Amazon Customer Support.

Queries the ChromaDB vector store, generates grounded @AmazonHelp-style
customer support replies, detects escalation triggers, and records
detailed latency and token telemetry.

Usage:
    python 2_rag_agent.py --query "Where is my refund?"
    python 2_rag_agent.py --interactive
    python 2_rag_agent.py --query "I will sue you" --top_k 5
"""

import os
import re
import sys
import time
import argparse
import chromadb
from chromadb.utils import embedding_functions
from src.llm_client import get_llm_client

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "amazon_support_kb"

ESCALATION_KEYWORDS = [
    "lawsuit", "lawyer", "legal", "chargeback", "fraud", "scam",
    "stolen", "unauthorized", "bbb", "ftc", "furious", "unacceptable",
    "angry", "worst", "horrible", "harassment",
]

SYSTEM_PROMPT = """You are an Amazon customer support agent operating under the @AmazonHelp handle.

Your persona:
- Empathetic, professional, direct, and concise.
- Always address the customer warmly and acknowledge their concern before providing a solution.
- Write in a natural, conversational tone as if you are directly messaging the customer.

Grounding rules:
- Base your answers ONLY on the provided context passages below. These are verified Amazon support policies and historical resolutions.
- NEVER fabricate refund amounts, return windows, delivery timelines, compensation offers, or any policies not explicitly stated in the context.
- If the provided context does not contain enough information to resolve the customer's issue, politely guide them to contact Amazon support directly via their account for secure, account-level assistance. Do not invent a solution.

Formatting rules:
- Do NOT include citation markers like [1], [2], or reference numbers in your response.
- Your response should read as a clean, natural customer support message.
- Keep your response concise — aim for 2-4 sentences unless the issue requires more detail.
"""

# ─── Vector Store Setup ───────────────────────────────────────────────────────

def get_collection():
    """Connect to the persistent ChromaDB collection with the matching embedding function."""
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
    )
    return collection

# ─── Retrieval Module ─────────────────────────────────────────────────────────

def retrieve_context(collection, query: str, top_k: int = 3) -> tuple:
    """
    Retrieve the top-K most relevant contexts from the vector store.

    Returns:
        tuple: (list[dict], float)
            - List of dicts with keys: text, tweet_id, distance
            - Retrieval latency in milliseconds
    """
    start = time.perf_counter()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    contexts = []
    if results and results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            ctx = {
                "text": doc,
                "tweet_id": results["metadatas"][0][i].get("tweet_id", "N/A"),
                "distance": results["distances"][0][i] if results.get("distances") else 0.0,
            }
            contexts.append(ctx)

    return contexts, latency_ms

# ─── Escalation Detection ────────────────────────────────────────────────────

def detect_escalation(query: str, response: str = "") -> bool:
    """
    Scan the customer query and generated response for high-risk
    escalation signals (legal threats, fraud, abusive language).
    """
    combined = (query + " " + response).lower()
    for keyword in ESCALATION_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', combined):
            return True
    return False

# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_user_prompt(query: str, contexts: list) -> str:
    """Build the user-facing prompt with retrieved context injected."""
    context_block = ""
    for i, ctx in enumerate(contexts, 1):
        context_block += f"Context {i} (Source: {ctx['tweet_id']}):\n{ctx['text']}\n\n"

    prompt = f"""--- Retrieved Policy Contexts ---
{context_block}
--- Customer Query ---
{query}

Based on the policy contexts above, provide a helpful response to the customer's query."""
    return prompt

# ─── Core Pipeline ────────────────────────────────────────────────────────────

def generate_response(query: str, top_k: int = 3, collection=None, llm_client=None) -> dict:
    """
    End-to-end RAG pipeline: retrieve → generate → escalation check → telemetry.

    Returns:
        dict: Standardized result with response, contexts, escalation flag, and metrics.
    """
    total_start = time.perf_counter()

    # Lazy-init shared resources if not injected
    if collection is None:
        collection = get_collection()
    if llm_client is None:
        llm_client = get_llm_client()

    # 1. Retrieval
    contexts, retrieval_latency_ms = retrieve_context(collection, query, top_k)

    # 2. Generation
    user_prompt = build_user_prompt(query, contexts)
    gen_start = time.perf_counter()
    try:
        llm_result = llm_client.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        generation_latency_ms = (time.perf_counter() - gen_start) * 1000
        # Unwrap tenacity RetryError to get the real cause
        root_cause = e.__cause__ if e.__cause__ else e
        error_msg = str(root_cause)
        # Detect quota/billing errors and give a clear message
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower() or "credit" in error_msg.lower():
            print(f"\n[ERROR] LLM API quota or rate limit exceeded.")
            print(f"  Provider: {type(llm_client).__name__} | Model: {llm_client.model_name}")
            print(f"  Details: {error_msg[:300]}")
            print(f"\n  Suggestions:")
            print(f"    - Wait a few minutes and retry (rate limit)")
            print(f"    - Add billing/credits to your API account (quota)")
            print(f"    - Switch provider via LLM_PROVIDER env var")
        else:
            print(f"\n[ERROR] LLM generation failed: {error_msg[:300]}")
        sys.exit(1)
    generation_latency_ms = (time.perf_counter() - gen_start) * 1000

    response_text = llm_result["text"]
    usage = llm_result["usage"]

    # 3. Escalation
    escalation = detect_escalation(query, response_text)

    # 4. Total latency
    total_latency_ms = (time.perf_counter() - total_start) * 1000

    return {
        "query": query,
        "response": response_text,
        "retrieved_contexts": [ctx["text"] for ctx in contexts],
        "source_tweet_ids": [ctx["tweet_id"] for ctx in contexts],
        "distances": [ctx["distance"] for ctx in contexts],
        "suggested_escalation": escalation,
        "metrics": {
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }

# ─── Formatted Output ────────────────────────────────────────────────────────

def print_result(result: dict):
    """Print a richly formatted terminal output block for a single query result."""
    m = result["metrics"]
    sep = "=" * 68

    print()
    print(sep)
    print("  CUSTOMER QUERY")
    print(sep)
    print(f"  {result['query']}")
    print()

    print("-- Retrieved Contexts " + "-" * 46)
    for i, ctx_text in enumerate(result["retrieved_contexts"], 1):
        tweet_id = result["source_tweet_ids"][i - 1] if i - 1 < len(result["source_tweet_ids"]) else "N/A"
        preview = ctx_text[:120].replace("\n", " ")
        if len(ctx_text) > 120:
            preview += "..."
        print(f"  [{i}] (Tweet: {tweet_id})")
        print(f"      {preview}")
        print()

    print("-- Generated Response " + "-" * 46)
    # Word-wrap the response for readability
    for line in result["response"].split("\n"):
        print(f"  {line}")
    print()

    esc_marker = "[!!] True -- ESCALATE TO HUMAN AGENT" if result["suggested_escalation"] else "[OK] False"
    print("-- Escalation " + "-" * 54)
    print(f"  Suggested Escalation: {esc_marker}")
    print()

    print("-- Telemetry " + "-" * 55)
    print(f"  Retrieval Latency:  {m['retrieval_latency_ms']:>10.2f} ms")
    print(f"  Generation Latency: {m['generation_latency_ms']:>10.2f} ms")
    print(f"  Total Latency:      {m['total_latency_ms']:>10.2f} ms")
    print(f"  Prompt Tokens:      {m['prompt_tokens']:>10}")
    print(f"  Completion Tokens:  {m['completion_tokens']:>10}")
    print(f"  Total Tokens:       {m['total_tokens']:>10}")
    print(sep)
    print()

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Amazon Customer Support RAG Agent — Query, Retrieve, Respond."
    )
    parser.add_argument("--query", type=str, default=None, help="Single customer query to process.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive query shell.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of contexts to retrieve (default: 3).")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_help()
        print("\nError: Provide --query or --interactive.")
        sys.exit(1)

    # Pre-initialize shared resources once
    print("Initializing vector store and LLM client...")
    collection = get_collection()
    llm_client = get_llm_client()
    print(f"  Vector store: {COLLECTION_NAME} ({collection.count()} documents)")
    print(f"  LLM provider: {type(llm_client).__name__} (model: {llm_client.model_name})")
    print()

    if args.query:
        result = generate_response(args.query, top_k=args.top_k, collection=collection, llm_client=llm_client)
        print_result(result)

    elif args.interactive:
        print("Interactive mode — type your query and press Enter. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                query = input("Customer> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting interactive mode.")
                break

            result = generate_response(query, top_k=args.top_k, collection=collection, llm_client=llm_client)
            print_result(result)


if __name__ == "__main__":
    main()
