# @AmazonHelp: Enterprise RAG Support Agent & Evaluation Console

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Console-Streamlit-red.svg)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

An end-to-end Retrieval-Augmented Generation (RAG) system modeled after Amazon's frontline `@AmazonHelp` operations. The platform automates customer query intake, retrieves verified policy context using local dense vector embeddings, performs multi-provider LLM response synthesis, enforces automated human escalation triggers, and runs full forensic audit benchmarking against 273 curated golden records with an automated LLM-as-a-Judge. 

Includes an enterprise light-themed observability and live resolution console inspired by Salesforce Service Cloud and Zoho Desk Zia with **zero external API key requirement for offline evaluator inspection**.

---

## 🏗️ Architecture & Pipeline Flow

```text
       [Inbound Customer Inquiry]
                   │
                   ▼
┌───────────────────────────────────────┐
│ 1. Vector Policy Retrieval (ChromaDB) │ ◄── Ingests 925 verified policies
│    Dense Cosine Similarity Search     │     Embeddings: all-MiniLM-L6-v2
└──────────────────┬────────────────────┘     Latency: ~21.4 ms
                   │ Top-K Chunks
                   ▼
┌───────────────────────────────────────┐
│ 2. Resolution Generation & Grounding  │ ◄── Multi-Provider Load Balancer
│    Empathetic @AmazonHelp Persona     │     (Groq / Gemini / OpenAI)
│    Strict Policy Context Constraints  │     Failover fallback on 429 quota
└──────────────────┬────────────────────┘
                   │ Synthesized Resolution
                   ▼
┌───────────────────────────────────────┐
│ 3. Automated Escalation & Safety Gate │ ◄── Frustration & Keyword Detector
│    Legal / Fraud / Chargeback Alerts  │     Rule-based safety triage
└──────────────────┬────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Direct Reply │      │ Human Escalate│
└───────┬──────┘      └───────┬──────┘
        │                     │
        ▼                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Evaluation & Forensic Observability Console                         │
│    • LLM-as-a-Judge harness over 273 benchmark records (94.9% Acc)     │
│    • Streamlit enterprise dashboard: KPIs, ticket audit, email dispatch│
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Evaluator Quickstart (< 3 Minutes)

Evaluators can inspect and verify the entire system offline immediately without creating or entering any API keys.

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/AI-support-agent.git
cd AI-support-agent

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. One-Time Vector DB Initialization (~10s)
*Required only if `chroma_db/` has not yet been indexed:*
```bash
python 1_index_kb.py
```
> Ingests and indexes 925 policy passages from `data/processed/production_rag_knowledge_base.csv` into `./chroma_db`.

### 3. Launch Observability Console (Zero API Key Needed)
```bash
streamlit run 4_dashboard.py
```
> **Instant Offline Mode**: Loads 273 forensic records directly from `data/reports/final_evaluation_report.json`.
> - **Executive KPI Dashboard**: Metrics, latency distributions, and intent breakdown charts.
> - **Forensic Audit Inspector**: Explore row-level prompts, gold intents, LLM answers, and judge rationales.
> - **Human-in-the-Loop Correction (MLOps)**: Modify agent responses directly in the inspector and export serialized `.jsonl` examples for LLM fine-tuning.
> - **Dual-Mode Inference Engine**: Switch seamlessly between **Offline Mode** (zero-cost benchmark replay) and **Live Cloud LLM Mode** (queries providers directly using keys in `.env`) directly from the UI sidebar.
> - **Live Copilot Console**: Run offline-curated or vector-grounded support queries and draft simulated email dispatches.

---

## 📁 Production Repository Layout

```text
AI-support-agent/
├── .streamlit/
│   └── config.toml                           # Custom Streamlit UI styling & server config
├── data/
│   ├── processed/
│   │   ├── production_rag_knowledge_base.csv # 925 scrubbed Amazon resolution policies
│   │   └── true_golden_eval_set.csv          # 290 stratified ground-truth benchmark queries
│   └── reports/
│       ├── eval_checkpoint.json              # State-checkpointed evaluation cache (273 items)
│       └── final_evaluation_report.json      # Aggregated metrics & forensic audit reports
├── src/
│   ├── __init__.py
│   └── llm_client.py                         # Multi-provider LLM abstraction (Round-Robin & Fallback)
├── chroma_db/                                # Persistent ChromaDB vector storage (all-MiniLM-L6-v2)
├── 1_index_kb.py                             # Vector indexing pipeline
├── 2_rag_agent.py                            # Core RAG generation and triage engine
├── 3_evaluate_system.py                      # LLM-as-a-Judge benchmark suite with atomic checkpointing
├── 4_dashboard.py                            # Streamlit enterprise operations & forensic console
├── requirements.txt                          # Pinned dependency requirements
├── .env.example                              # Environment variable configuration template
├── .gitignore                                # Version control exclusion rules
├── PROJECT_CONTEXT.md                        # Architecture & data schema reference
└── README.md
```

---

## 🛠️ Step-by-Step Pipeline Usage

### Step 1: Ingest & Index Knowledge Base
Embeds the clean resolution policies into ChromaDB with cosine similarity indexing:
```bash
python 1_index_kb.py
```
*Options:*
- `--data_path`: Custom dataset path (default: `./data/processed/production_rag_knowledge_base.csv`).
- `--reset`: Reset and rebuild the Chroma collection.
- `--batch_size`: Embedding batch size (default: `50`).

---

### Step 2: Query the Support Agent (CLI & Interactive)
Run the grounded `@AmazonHelp` agent with contextual retrieval and triage:

```bash
# Direct CLI query:
python 2_rag_agent.py --query "Where is my refund for order 104-9283748-1294829?"

# Interactive customer simulation terminal:
python 2_rag_agent.py --interactive

# Retrieve top 5 policy passages:
python 2_rag_agent.py --query "I was charged twice for Prime" --top_k 5
```

---

### Step 3: Run the Automated Evaluation Suite
Runs stateful LLM-as-a-Judge evaluation across ground-truth queries with automatic progress checkpointing:

```bash
# Smoke test (5 random samples):
python 3_evaluate_system.py --sample 5 --delay 0.5

# Slice-based evaluation (rows 0 through 50):
python 3_evaluate_system.py --start-idx 0 --end-idx 50 --delay 0.5

# Full evaluation across all golden records:
python 3_evaluate_system.py --delay 0.5

# Reset checkpoint to evaluate from scratch:
python 3_evaluate_system.py --reset-checkpoint --sample 10 --delay 0.5
```

*Results are automatically written to `data/reports/final_evaluation_report.json` and cached in `data/reports/eval_checkpoint.json`.*

---

## 📊 Benchmark Results & Evaluation Rubric

The system was evaluated against 273 stratified ground-truth customer queries using an automated LLM-as-a-Judge quality harness:

### Core Performance Metrics

| Metric Dimension | Score | Benchmark Target | Description |
| :--- | :---: | :---: | :--- |
| **Faithfulness & Groundedness** | **4.95 / 5.0** | > 4.50 | Absence of hallucinated policies, timelines, or refund amounts |
| **Helpfulness & Persona Tone** | **4.91 / 5.0** | > 4.50 | Warm, empathetic, concise `@AmazonHelp` operational style |
| **Intent Alignment** | **4.76 / 5.0** | > 4.50 | Response directly addresses the customer's core query |
| **Composite Quality Score** | **4.88 / 5.0** | > 4.50 | Harmonic overall answer quality rating |
| **Escalation Accuracy** | **94.87%** | > 90.0% | Safety gate classification accuracy |
| **Escalation Precision** | **87.5%** | > 80.0% | Minimized false-positive human escalations |
| **Mean Retrieval Latency** | **21.38 ms** | < 100 ms | Local vector similarity search speed |

### Performance Breakdown by Customer Intent

| Customer Intent Category | Records | Intent Alignment | Faithfulness | Tone / Persona |
| :--- | :---: | :---: | :---: | :---: |
| **Order Tracking & Delay** | 66 | 4.91 / 5.0 | 4.95 / 5.0 | 4.95 / 5.0 |
| **Subscription & Billing (Prime)** | 58 | 4.64 / 5.0 | 4.97 / 5.0 | 4.93 / 5.0 |
| **Account Access & General Inquiry** | 70 | 4.83 / 5.0 | 4.94 / 5.0 | 4.87 / 5.0 |
| **Refund & Return Request** | 66 | 4.68 / 5.0 | 4.95 / 5.0 | 4.88 / 5.0 |
| **Damaged or Missing Item** | 13 | 4.62 / 5.0 | 4.92 / 5.0 | 4.85 / 5.0 |

---

## 🔄 Multi-Provider Resilience & Failover

The LLM abstraction layer in `src/llm_client.py` ensures continuous enterprise availability:

- **Round-Robin Cycling**: Cycles dynamically across configured providers in chunks of 5 requests (`ROUND_ROBIN_CHUNK_SIZE=5`).
- **Synchronized State**: Shared thread-safe counters balance load across both agent inference and LLM-as-a-Judge scoring.
- **Failover on Rate Limits**: Automatically catches HTTP 429 / quota exhaustion errors and transparently routes the pending request to the next available provider in the pool (Groq ➔ Gemini ➔ OpenAI).
- **Environment Flexibility**: Set keys via `.env` (copy from `.env.example`):
  ```env
  GROQ_API_KEY=gsk_...
  GEMINI_API_KEY=AIza...
  OPENAI_API_KEY=sk-proj-...
  LLM_PROVIDER=groq
  LLM_MODEL_NAME=llama-3.3-70b-versatile
  ```

---

## 🔒 Security & Privacy

- **Zero Data Leakage**: Sensitive credentials (`.env`, `*.key`, `secrets/`) are strictly excluded from version control.
- **Customer Privacy**: Customer handles, account IDs, and identifiable tokens were stripped and sanitized to `[CUSTOMER]`.
- **Local Dense Search**: Vector embeddings and cosine retrieval execute entirely locally on device via `all-MiniLM-L6-v2` with zero external document transmission.

---

## 7. Attributions, Citations & Data Provenance

This project leverages open-source data, pre-trained embeddings, and external foundational models. Every imported dependency, baseline dataset, and model artifact is documented below alongside the specific engineering transformations applied by our team.

### Data Sources & Curation Provenance

| Asset | Source / Origin | Purpose in Pipeline | Modifications & Engineering Added |
| :--- | :--- | :--- | :--- |
| **Raw Customer Support Interactions** | [Kaggle Customer Support on Twitter (`twcs.csv`)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | Source corpus filtered for `@AmazonHelp` interactions. | Ingested 700 inbound queries and 1,000 outbound agent responses. Applied ASCII encoding filters (`> 0.85`), scrubbed `[http://t.co/](http://t.co/)...` URLs, anonymized usernames to `[CUSTOMER]`, and stripped repetitive corporate boilerplate (e.g., *"Please DM us"*) via custom regex. |
| **Golden Evaluation Dataset (`true_golden_eval_set.csv`)** | Curated derived subset ($N=290$) | Benchmark dataset for evaluating intent accuracy, faithfulness, and escalation precision. | Implemented stratified sampling with mathematical capping at 72 rows per dominant class to eliminate a 72% generic inquiry bias. Retained 100% of minority edge cases (e.g., all 13 damaged item reports) and flagged safety risk keywords (`fraud`, `lawsuit`). |
| **RAG Knowledge Base (`production_rag_knowledge_base.csv`)** | Sanitized derived corpus ($N=925$) | Contextual ground truth indexed into vector storage for grounded retrieval. | Vectorized into ChromaDB with metadata attributes to support runtime cosine similarity lookups. |

### Models & External Libraries

| Component | Provider / Library | Artifact / Version | Implementation Role |
| :--- | :--- | :--- | :--- |
| **Dense Vector Embeddings** | Hugging Face / Sentence-Transformers | `all-MiniLM-L6-v2` | Computes local 384-dimensional embeddings for knowledge base chunking and query retrieval, operating with sub-25ms latency. |
| **Vector Storage Engine** | Chroma | `chromadb` | Embedded local vector database for cosine similarity search. |
| **Inference & Evaluation LLMs** | Multi-Provider API (Google, Groq, OpenAI) | `gemini-2.5-flash`, `gpt-4o-mini`, Llama 3 / Mixtral endpoints | Multi-provider round-robin routing abstraction (`src/llm_client.py`) with exponential backoff and stateful checkpointing (`eval_checkpoint.json`) to bypass HTTP 429 rate limits. |
| **Frontend Observability Console** | Streamlit | `streamlit` | Powers the Executive Analytics dashboard, Forensic Split-View Inspector, and Interactive Copilot interface. |

### Conceptual Attribution

* **Evaluation Architecture:** The LLM-as-a-Judge methodology used in `3_evaluate_system.py` adopts the evaluation paradigm popularized by Ragas and G-Eval (scoring on Faithfulness, Intent Alignment, and Persona Adherence on a 1–5 scale). Custom evaluation rubrics were designed specifically for tier-1 retail support, supported by a 50-sample human spot-check validation demonstrating 94% score concordance.

---

## 📜 License

This project is open-source under the [Apache 2.0 License](LICENSE).
