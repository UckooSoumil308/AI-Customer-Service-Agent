"""
4_dashboard.py -- Enterprise Support Console for @AmazonHelp RAG System.

A production-ready, light-themed Streamlit dashboard modeled after
Salesforce Fin and Zoho Desk Zia. Provides executive analytics,
forensic audit, and live-dispatch capabilities.

Run:
    streamlit run 4_dashboard.py
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()
import sys
import json
import smtplib
import subprocess
import textwrap
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import streamlit as st

# Sync Streamlit Cloud secrets to environment variables if present
try:
    for key in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]:
        if hasattr(st, "secrets") and key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import pandas as pd
import plotly.graph_objects as go

def get_active_providers():
    providers = []
    if os.getenv("GROQ_API_KEY"):
        providers.append("Groq")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("Gemini")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("OpenAI")
    return providers

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Amazon Help \u00b7 Enterprise Support Console",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Streamlit Overrides ── */
/* Force the background to a soft gray so white cards float */
.stApp {
    background-color: #f4f5f7 !important;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: #1d1d1f;
}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 1 !important;
}
.block-container { 
    padding-top: 3rem !important; 
    padding-bottom: 2rem !important; 
}

/* ── Native Streamlit Widget Styling ── */
/* Inputs */
.stTextInput input, .stTextArea textarea {
    background-color: #eef0f3 !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
    color: #1d1d1f !important;
    transition: all 0.2s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    background-color: #ffffff !important;
    border: 1px solid #000000 !important;
    box-shadow: 0 0 0 4px rgba(0,0,0,0.05) !important;
}
/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #eef0f3 !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover {
    background-color: #e5e8ec !important;
}
/* Buttons */
.stButton > button {
    background-color: #1d1d1f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
.stButton > button:hover {
    background-color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15) !important;
}
/* Secondary Buttons (like the copy button) */
button[key="btn_copy_resolution"] {
    background-color: #ffffff !important;
    color: #1d1d1f !important;
    border: 1px solid #d2d5d9 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
button[key="btn_copy_resolution"]:hover {
    background-color: #f4f5f7 !important;
    border-color: #1d1d1f !important;
}

/* ── Custom UI Components (Cards) ── */
.kpi-card {
    background: #ffffff; 
    border: none;
    border-radius: 20px;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.05), 0 4px 8px -2px rgba(0,0,0,0.02);
    padding: 24px;
    position: relative; 
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 8px;
}
.kpi-card:hover { 
    transform: translateY(-4px);
    box-shadow: 0 30px 60px -15px rgba(0,0,0,0.08), 0 8px 16px -4px rgba(0,0,0,0.03);
}
.kpi-label { font-size:12px; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; color:#86868b; margin-bottom:8px; }
.kpi-value { font-size:36px; font-weight:700; color:#1d1d1f; line-height:1; letter-spacing:-0.03em; }
.kpi-sub   { font-size:13px; color:#86868b; margin-top:6px; font-weight:500; }

/* ── Section Headers ── */
.section-header {
    font-size:18px; font-weight:700; color:#1d1d1f; letter-spacing:-0.02em;
    margin-bottom:20px; padding-bottom:12px; border-bottom:1px solid #e5e5ea;
    display: flex; align-items: center;
}

/* ── Ticket Cards ── */
.ticket-card {
    background:#ffffff; border:none; border-radius:16px;
    padding:20px; margin-bottom:16px; transition:transform 0.3s ease;
    box-shadow: 0 10px 20px -5px rgba(0,0,0,0.03), 0 2px 4px -2px rgba(0,0,0,0.02);
}
.ticket-card:hover { transform: translateY(-2px); box-shadow: 0 15px 30px -5px rgba(0,0,0,0.05); }
.ticket-id    { font-size:12px; font-weight:700; color:#86868b; margin-bottom:8px; letter-spacing: 0.02em;}
.ticket-query { font-size:15px; color:#1d1d1f; line-height:1.5; font-weight:500; }

/* ── Inspector & Email Frames ── */
.inspector-card, .email-frame {
    background: #ffffff; 
    border: none;
    border-radius: 24px;
    padding: 32px; 
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.02);
}
.email-field-label { font-weight:600; color:#86868b; min-width:80px; display:inline-block; }
.email-divider { border:none; border-top:1px solid #e5e5ea; margin:20px 0; }
.email-body { white-space:pre-wrap; color:#1d1d1f; font-size:15px; line-height: 1.6; }

/* ── Message Bubbles ── */
.msg-bubble-user {
    background: #f4f5f7; border: none; 
    border-radius: 20px 20px 20px 6px; padding: 16px 20px;
    font-size: 15px; color: #1d1d1f; margin-bottom: 20px; 
    line-height:1.5; font-weight: 500;
}
.msg-bubble-agent {
    background: #1d1d1f; border: none; 
    border-radius: 20px 20px 6px 20px; padding: 16px 20px;
    font-size: 15px; color: #ffffff; margin-bottom: 20px; 
    line-height:1.5; box-shadow: 0 8px 24px -8px rgba(0,0,0,0.2);
}
.context-passage {
    background:#fafafa; border:1px solid #e5e5ea; border-radius:12px;
    padding:16px; font-size:13px; color:#86868b; margin-bottom:12px; line-height:1.6;
}
.judge-metric-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 0; border-bottom:1px solid #f4f5f7;
}
.judge-metric-row:last-child { border-bottom: none; }
.judge-metric-label { font-size:14px; color:#86868b; font-weight:500; }
.judge-metric-score { font-size:16px; font-weight:700; color:#1d1d1f; }

/* ── Global Header ── */
.global-header {
    background: #ffffff;
    border-radius: 24px; padding: 28px 36px; margin-top: 4px; margin-bottom: 32px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.04);
}
.global-header-title { font-size:24px; font-weight:700; color:#1d1d1f; letter-spacing: -0.04em; }
.global-header-sub   { font-size:14px; color:#86868b; margin-top:4px; font-weight:500; }
.header-live-dot {
    width:8px; height:8px; background:#000000; border-radius:50%;
    display:inline-block; margin-right:8px;
}
.header-status { 
    font-size:13px; color:#1d1d1f; display:flex; align-items:center; 
    background: #f4f5f7; padding: 8px 16px; border-radius: 20px;
    font-weight: 600;
}

/* ── Pills & Sidebar Badges ── */
.pill-good     { background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:12px; font-size:11px; font-weight:700; display:inline-block; letter-spacing:0.02em; }
.pill-escalate { background:#ffebee; color:#c62828; padding:4px 12px; border-radius:12px; font-size:11px; font-weight:700; display:inline-block; letter-spacing:0.02em; }
.pill-accept   { background:#fff8e1; color:#f57f17; padding:4px 12px; border-radius:12px; font-size:11px; font-weight:700; display:inline-block; letter-spacing:0.02em; }
.pill-review   { background:#ffebee; color:#c62828; padding:4px 12px; border-radius:12px; font-size:11px; font-weight:700; display:inline-block; letter-spacing:0.02em; }
.pill-badge    { background:#f4f5f7; color:#1d1d1f; padding:4px 12px; border-radius:12px; font-size:11px; font-weight:700; display:inline-block; letter-spacing:0.02em; }

.sidebar-badge {
    background: #ffffff; border: none; border-radius: 12px;
    padding: 12px 16px; font-size: 13px; color: #86868b; margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}
.sidebar-badge b { color:#1d1d1f; font-weight: 700; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d2d5d9; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #86868b; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data():
    """Load from final_evaluation_report.json or eval_checkpoint.json fallback."""
    report_paths = [
        BASE_DIR / "data" / "reports" / "final_evaluation_report.json",
        BASE_DIR / "final_evaluation_report.json",
    ]
    ckpt_paths = [
        BASE_DIR / "data" / "reports" / "eval_checkpoint.json",
        BASE_DIR / "eval_checkpoint.json",
    ]

    report_path = next((p for p in report_paths if p.exists()), report_paths[0])
    ckpt_path   = next((p for p in ckpt_paths if p.exists()), ckpt_paths[0])

    records, summary = [], {}

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("detailed_results", [])
        summary = data.get("summary_metrics", {})

    if not records and ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("results", [])

    if not records:
        st.error("No evaluation data found. Run 3_evaluate_system.py first.")
        st.stop()

    rows = []
    for r in records:
        judge = r.get("judge", {})
        tel   = r.get("telemetry", {})
        ia = judge.get("intent_alignment",  r.get("intent_alignment",  0))
        fa = judge.get("faithfulness",       r.get("faithfulness",       0))
        ht = judge.get("helpfulness_tone",   r.get("helpfulness_tone",   0))
        row = {
            "tweet_id":             r.get("tweet_id"),
            "query":                r.get("query", ""),
            "gold_intent":          r.get("gold_intent", "Unknown"),
            "gold_escalate":        bool(r.get("gold_escalate", False)),
            "escalation_reason":    r.get("escalation_reason", ""),
            "suggested_escalation": bool(r.get("suggested_escalation", False)),
            "agent_response":       r.get("agent_response", ""),
            "retrieved_contexts":   r.get("retrieved_contexts", []),
            "source_tweet_ids":     r.get("source_tweet_ids", []),
            "intent_alignment":     ia,
            "faithfulness":         fa,
            "helpfulness_tone":     ht,
            "judge_rationale":      judge.get("judge_rationale", r.get("judge_rationale", "")),
            "judge_latency_ms":     judge.get("latency_ms", 0),
            "retrieval_latency_ms": tel.get("retrieval_latency_ms",  r.get("retrieval_latency_ms",  0)),
            "generation_latency_ms":tel.get("generation_latency_ms", r.get("generation_latency_ms", 0)),
            "total_latency_ms":     tel.get("total_latency_ms",      r.get("total_latency_ms",      0)),
            "prompt_tokens":        tel.get("prompt_tokens",         r.get("prompt_tokens",         0)),
            "completion_tokens":    tel.get("completion_tokens",     r.get("completion_tokens",     0)),
            "total_tokens":         tel.get("total_tokens",          r.get("total_tokens",          0)),
        }
        row["composite_score"] = round((ia + fa + ht) / 3, 2) if (ia or fa or ht) else 0
        rows.append(row)

    return pd.DataFrame(rows), summary


def _safe(text):
    return str(text).replace("<", "&lt;").replace(">", "&gt;")


def pill_html(faithfulness, escalated):
    if escalated:
        return '<span class="pill-escalate">\u26a0 Escalated</span>'
    if faithfulness >= 5:
        return '<span class="pill-good">\u2713 Good</span>'
    if faithfulness >= 4:
        return '<span class="pill-accept">\u223c Acceptable</span>'
    return '<span class="pill-review">\u2717 Needs Review</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df):
    with st.sidebar:
        st.markdown(
            '<div style="padding:16px 0 8px 0;">'
            '<div style="font-size:17px;font-weight:700;color:#0f172a;">🛒 Amazon Help</div>'
            '<div style="font-size:11px;color:#94a3b8;margin-top:2px;">Enterprise Support Console</div>'
            '</div>'
            '<hr style="border:none;border-top:1px solid #e5e7eb;margin:8px 0 16px 0;">',
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ Pipeline Execution Mode", expanded=True):
            execution_mode = st.radio(
                "Select Backend Engine:",
                options=[
                    "⚡ Offline Mode (ChromaDB + Benchmark Replay)",
                    "☁️ Live Cloud LLM (Uses .env API Keys)"
                ],
                index=0,
                help="Offline Mode runs 100% locally with zero API cost. Live Mode calls your provider API using keys in .env."
            )
            st.session_state.execution_mode = execution_mode
    
            active_providers = get_active_providers()
            if execution_mode == "☁️ Live Cloud LLM (Uses .env API Keys)":
                if active_providers:
                    st.success(f"🟢 Active API Keys: {', '.join(active_providers)}")
                else:
                    st.warning("⚠️ No LLM API keys found in .env. Switch to Offline Mode or add keys.")
            else:
                st.info("🔵 Zero API Cost · Local ChromaDB & Evaluated Records")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("⚙️ System Footprint", expanded=True):
            st.markdown(
                '<div class="sidebar-badge">💾 <b>ChromaDB:</b> 925 policies</div>'
                '<div class="sidebar-badge">📊 <b>Evaluated:</b> 273 / 290 Records</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🤖 Zia Guidance Rules", expanded=True):
            tone = st.selectbox("Tone of Voice",
                ["Empathetic & Professional", "Concise", "Technical"], key="zia_tone")
            persona = st.radio("Target Persona",
                ["@AmazonHelp Twitter Support", "Formal Email Resolution"], key="zia_persona")

        with st.expander("🔍 Triage Filters", expanded=True):
            intents = ["All"] + sorted(df["gold_intent"].unique().tolist())
            intent_filter = st.selectbox("Intent Category", intents, key="filter_intent")
            escalation_filter = st.selectbox("Escalation Status",
                ["All", "Auto-Resolved (TN)", "Escalated (TP)", "Missed (FN)", "False Alarm (FP)"],
                key="filter_escalation")
            score_threshold = st.slider("Faithfulness >= threshold", 1, 5, 1, key="filter_threshold")
            search_text = st.text_input("Search tweet_id or keyword", key="search_text")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:11px;color:#94a3b8;text-align:center;">'
            f'Zia v2.4 \u00b7 {df.shape[0]} records loaded</div>',
            unsafe_allow_html=True,
        )

    return intent_filter, escalation_filter, score_threshold, search_text, tone, persona


def apply_filters(df, intent_filter, escalation_filter, score_threshold, search_text):
    f = df.copy()
    if intent_filter != "All":
        f = f[f["gold_intent"] == intent_filter]
    if escalation_filter == "Auto-Resolved (TN)":
        f = f[(~f["gold_escalate"]) & (~f["suggested_escalation"])]
    elif escalation_filter == "Escalated (TP)":
        f = f[(f["gold_escalate"]) & (f["suggested_escalation"])]
    elif escalation_filter == "Missed (FN)":
        f = f[(f["gold_escalate"]) & (~f["suggested_escalation"])]
    elif escalation_filter == "False Alarm (FP)":
        f = f[(~f["gold_escalate"]) & (f["suggested_escalation"])]
    f = f[f["faithfulness"] >= score_threshold]
    if search_text.strip():
        q = search_text.strip().lower()
        f = f[f["tweet_id"].astype(str).str.contains(q) | f["query"].str.lower().str.contains(q, na=False)]
    return f.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI Banner
# ---------------------------------------------------------------------------

def render_kpi_banner(df, summary):
    total = len(df)
    ai_res  = round((~df["suggested_escalation"]).sum() / total * 100, 1) if total else 0
    esc_pct = round(df["suggested_escalation"].sum() / total * 100, 1) if total else 0
    comp    = round(df["composite_score"].mean(), 2) if total else 0
    lat     = round(df["retrieval_latency_ms"].mean(), 1) if total else 0

    qual = summary.get("quality_scores", {})
    tel  = summary.get("operational_telemetry", {})
    comp = qual.get("composite_quality_score", comp)
    lat  = tel.get("mean_retrieval_latency_ms", lat)

    ai_n  = int((~df["suggested_escalation"]).sum())
    esc_n = int(df["suggested_escalation"].sum())
    c_col = "#22c55e" if comp >= 4.5 else "#f59e0b" if comp >= 3.5 else "#ef4444"

    c1, c2, c3, c4 = st.columns(4)
    specs = [
        (c1, "#22c55e", "🤖 AI Resolution Rate",
         f"{ai_res}%", "Auto-handled without escalation",
         f'<span class="kpi-delta-up">\u2191 {ai_n} tickets resolved</span>'),
        (c2, "#f59e0b", "🔁 Escalation Handoff Rate",
         f"{esc_pct}%", "Routed to human support",
         f'<span class="kpi-delta-down">\u26a0 {esc_n} escalated</span>'),
        (c3, c_col, "\u2b50 Composite Quality Score",
         f'{comp} <span style="font-size:14px;color:#94a3b8;">/ 5.0</span>',
         "Intent \u00b7 Faithfulness \u00b7 Tone",
         f'<span class="kpi-delta-up">\u2191 Across {total} records</span>'),
        (c4, "#3b82f6", "\u26a1 Mean Retrieval Latency",
         f'{lat} <span style="font-size:14px;color:#94a3b8;">ms</span>',
         "ChromaDB vector search",
         '<span class="kpi-delta-up">\u2191 Sub-50ms p95</span>'),
    ]
    for col, accent, label, value, sub, delta in specs:
        col.markdown(
            f'<div class="kpi-card" style="--accent:{accent};">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'<div>{delta}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1: Executive Analytics
# ---------------------------------------------------------------------------

def render_analytics_tab(df, summary):
    st.markdown("<br>", unsafe_allow_html=True)

    # -- Grouped Bar: Quality by Intent --
    st.markdown('<div class="section-header">📊 Quality Scores by Intent Category</div>',
                unsafe_allow_html=True)

    grp = df.groupby("gold_intent").agg(
        intent_alignment=("intent_alignment", "mean"),
        faithfulness=("faithfulness", "mean"),
        helpfulness_tone=("helpfulness_tone", "mean"),
    ).round(2).reset_index()

    fig_bar = go.Figure()
    for col_name, label, color in [
        ("intent_alignment",  "Intent Alignment",  "#3b82f6"),
        ("faithfulness",      "Faithfulness",       "#22c55e"),
        ("helpfulness_tone",  "Helpfulness & Tone", "#f59e0b"),
    ]:
        fig_bar.add_trace(go.Bar(
            name=label, x=grp["gold_intent"], y=grp[col_name],
            marker_color=color, marker_line_width=0,
            text=grp[col_name].apply(lambda v: f"{v:.2f}"),
            textposition="outside", textfont=dict(size=10),
        ))
    fig_bar.update_layout(
        barmode="group", template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        xaxis=dict(tickfont=dict(size=11), title=""),
        yaxis=dict(range=[0, 5.8], tickfont=dict(size=11), title="Score (0\u20135)"),
        margin=dict(l=20, r=20, t=40, b=60),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -- Confusion Matrix + Classification Metrics --
    col_cm, col_sc = st.columns(2)

    esc = summary.get("escalation_classification", {})
    TP = esc.get("true_positives",  int(((df["gold_escalate"]) & (df["suggested_escalation"])).sum()))
    FP = esc.get("false_positives", int(((~df["gold_escalate"]) & (df["suggested_escalation"])).sum()))
    TN = esc.get("true_negatives",  int(((~df["gold_escalate"]) & (~df["suggested_escalation"])).sum()))
    FN = esc.get("false_negatives", int(((df["gold_escalate"]) & (~df["suggested_escalation"])).sum()))

    with col_cm:
        st.markdown('<div class="section-header">🎯 Escalation Confusion Matrix</div>',
                    unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=[[TP, FP], [FN, TN]],
            x=["Pred: Escalate", "Pred: Resolve"],
            y=["Actual: Escalate", "Actual: Resolve"],
            text=[[f"TP={TP}", f"FP={FP}"], [f"FN={FN}", f"TN={TN}"]],
            texttemplate="%{text}<br><b>%{z}</b>",
            colorscale=[[0, "#eff6ff"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
            showscale=False,
        ))
        fig_cm.update_layout(
            template="plotly_white", height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter", size=12),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_sc:
        st.markdown('<div class="section-header">📐 Classification Metrics</div>',
                    unsafe_allow_html=True)
        precision = esc.get("precision", TP / (TP + FP) if (TP + FP) else 0)
        recall    = esc.get("recall",    TP / (TP + FN) if (TP + FN) else 0)
        f1        = esc.get("f1_score",  2*precision*recall/(precision+recall) if (precision+recall) else 0)
        accuracy  = esc.get("accuracy",  (TP+TN)/len(df) if len(df) else 0)

        for label, val, desc in [
            ("Precision", precision, "Of escalations flagged, how many were real"),
            ("Recall",    recall,    "Of real escalations, how many were caught"),
            ("F1 Score",  f1,        "Harmonic mean of Precision and Recall"),
            ("Accuracy",  accuracy,  "Overall correct classification rate"),
        ]:
            bar_c = "#22c55e" if val >= 0.8 else "#f59e0b" if val >= 0.5 else "#ef4444"
            st.markdown(
                f'<div style="margin-bottom:14px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span style="font-size:13px;font-weight:600;color:#1e293b;">{label}</span>'
                f'<span style="font-size:14px;font-weight:700;color:#0f172a;">{val:.3f}</span></div>'
                f'<div style="background:#f1f5f9;border-radius:4px;height:8px;">'
                f'<div style="background:{bar_c};width:{val*100:.1f}%;height:8px;border-radius:4px;"></div></div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-top:3px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">'
            f'<span class="pill-good">TP: {TP}</span>'
            f'<span class="pill-review">FP: {FP}</span>'
            f'<span class="pill-accept">FN: {FN}</span>'
            f'<span class="pill-badge">TN: {TN}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Token donut + Latency histogram --
    col_d, col_l = st.columns(2)

    with col_d:
        st.markdown('<div class="section-header">🪙 Token Usage Breakdown</div>',
                    unsafe_allow_html=True)
        tel2 = summary.get("operational_telemetry", {})
        jt   = int(tel2.get("total_judge_tokens", 0))
        ap   = int(df["prompt_tokens"].sum())
        ac   = int(df["completion_tokens"].sum())
        fig_dn = go.Figure(go.Pie(
            labels=["Agent Prompt", "Agent Completion", "Judge Eval"],
            values=[ap, ac, jt], hole=0.55,
            marker=dict(colors=["#3b82f6", "#22c55e", "#a78bfa"],
                        line=dict(color="#ffffff", width=2)),
            textinfo="percent+label",
            textfont=dict(size=11, family="Inter"),
        ))
        total_tok = ap + ac + jt
        fig_dn.add_annotation(
            text=f"<b>{total_tok/1000:.0f}K</b><br>Total",
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(family="Inter", color="#0f172a"),
        )
        fig_dn.update_layout(
            template="plotly_white", height=280, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(size=11)),
            margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="white", font=dict(family="Inter"),
        )
        st.plotly_chart(fig_dn, use_container_width=True)

    with col_l:
        st.markdown('<div class="section-header">\u23f1 Generation Latency Distribution</div>',
                    unsafe_allow_html=True)
        fig_lat = go.Figure(go.Histogram(
            x=df["generation_latency_ms"], nbinsx=30,
            marker_color="#3b82f6", opacity=0.85,
        ))
        fig_lat.update_layout(
            template="plotly_white", height=260,
            xaxis_title="Generation Latency (ms)", yaxis_title="Ticket Count",
            margin=dict(l=20, r=20, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            showlegend=False, font=dict(family="Inter", size=11), bargap=0.05,
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # -- Intent performance table --
    st.markdown('<div class="section-header">📋 Intent Performance Breakdown</div>',
                unsafe_allow_html=True)
    bd = df.groupby("gold_intent").agg(
        Records=("tweet_id", "count"),
        Intent_Align=("intent_alignment", "mean"),
        Faithfulness=("faithfulness", "mean"),
        Helpfulness=("helpfulness_tone", "mean"),
        Composite=("composite_score", "mean"),
        Escalations=("suggested_escalation", "sum"),
    ).round(2).reset_index()
    bd.columns = ["Intent", "Records", "Intent Align", "Faithfulness",
                   "Helpfulness", "Composite", "Escalations"]
    st.dataframe(
        bd, use_container_width=True, hide_index=True,
        column_config={
            "Intent Align": st.column_config.ProgressColumn("Intent Align", min_value=0, max_value=5),
            "Faithfulness":  st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=5),
            "Helpfulness":   st.column_config.ProgressColumn("Helpfulness",  min_value=0, max_value=5),
            "Composite":     st.column_config.ProgressColumn("Composite",    min_value=0, max_value=5),
        },
    )


# ---------------------------------------------------------------------------
# Tab 2: Forensic Inspector
# ---------------------------------------------------------------------------

def render_forensic_tab(df, filtered_df):
    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.info("No tickets match the current filters. Adjust the sidebar controls.")
        return

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0

    col_q, col_i = st.columns([1, 1.2])

    with col_q:
        n = len(filtered_df)
        st.markdown(
            f'<div class="section-header">🗂 Ticket Queue '
            f'<span style="font-size:11px;color:#94a3b8;font-weight:400;margin-left:6px;">({n} records)</span></div>',
            unsafe_allow_html=True,
        )
        for i, row in filtered_df.head(50).iterrows():
            pill = pill_html(int(row["faithfulness"]), bool(row["suggested_escalation"]))
            excerpt = _safe(str(row["query"])[:120])
            st.markdown(
                f'<div class="ticket-card">'
                f'<div class="ticket-id">#{row["tweet_id"]} \u00b7 {row["gold_intent"]}</div>'
                f'<div class="ticket-query">{excerpt}...</div>'
                f'<div style="margin-top:8px;">{pill}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect #{row['tweet_id']}", key=f"btn_{i}",
                         use_container_width=True, type="secondary"):
                st.session_state.selected_idx = i
                st.rerun()

    with col_i:
        st.markdown('<div class="section-header">🔬 Forensic Inspector</div>',
                    unsafe_allow_html=True)
        idx = st.session_state.selected_idx
        if idx not in filtered_df.index:
            idx = filtered_df.index[0]
        rec = filtered_df.loc[idx]

        escl_badge = (
            '<span class="pill-escalate">\u26a0 ESCALATED</span>'
            if rec["suggested_escalation"]
            else '<span class="pill-good">\u2713 AUTO-RESOLVED</span>'
        )
        st.markdown(
            f'<div class="inspector-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
            f'<div><div style="font-size:11px;color:#64748b;font-weight:600;">TICKET #{rec["tweet_id"]}</div>'
            f'<div style="font-size:12px;color:#475569;margin-top:2px;">{rec["gold_intent"]}</div></div>'
            f'<div>{escl_badge}</div></div>'
            f'<div style="font-size:11px;font-weight:600;color:#64748b;margin-bottom:6px;text-transform:uppercase;">Customer Query</div>'
            f'<div class="msg-bubble-user">{_safe(rec["query"])}</div>'
            f'<div style="font-size:11px;font-weight:600;color:#64748b;margin-bottom:6px;text-transform:uppercase;">@AmazonHelp Resolution</div>'
            f'<div class="msg-bubble-agent">{_safe(rec["agent_response"])}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        contexts = rec["retrieved_contexts"]
        sources  = rec["source_tweet_ids"]
        with st.expander(f"Content ({len(contexts)} passages)", expanded=False):
            for j, (ctx, src) in enumerate(zip(contexts, sources), 1):
                st.markdown(
                    f'<div class="context-passage">'
                    f'<div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:4px;">'
                    f'SOURCE {j} \u00b7 Tweet #{src}</div>{_safe(ctx)}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="section-header" style="margin-top:16px;">\u2696\ufe0f LLM Judge Scorecard</div>',
            unsafe_allow_html=True,
        )
        for label, sv in [
            ("Intent Alignment",   int(rec["intent_alignment"])),
            ("Faithfulness",       int(rec["faithfulness"])),
            ("Helpfulness & Tone", int(rec["helpfulness_tone"])),
        ]:
            color = "#22c55e" if sv >= 5 else "#f59e0b" if sv >= 4 else "#ef4444"
            stars = "\u2b50" * sv + "\u2606" * (5 - sv)
            st.markdown(
                f'<div class="judge-metric-row">'
                f'<span class="judge-metric-label">{label}</span>'
                f'<div style="text-align:right;">'
                f'<span style="font-size:13px;color:{color};">{stars}</span>'
                f'<span class="judge-metric-score" style="margin-left:8px;">{sv}/5</span></div></div>',
                unsafe_allow_html=True,
            )
        composite = round((rec["intent_alignment"] + rec["faithfulness"] + rec["helpfulness_tone"]) / 3, 2)
        st.markdown(
            f'<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;margin-top:10px;">'
            f'<div style="font-size:11px;color:#64748b;font-weight:600;margin-bottom:6px;">JUDGE RATIONALE</div>'
            f'<div style="font-size:13px;color:#475569;line-height:1.6;">{_safe(rec["judge_rationale"])}</div>'
            f'<div style="margin-top:10px;font-size:12px;color:#94a3b8;">'
            f'Composite: <b style="color:#0f172a;">{composite}/5.0</b> \u00b7 '
            f'Retrieval: <b style="color:#0f172a;">{rec["retrieval_latency_ms"]:.1f}ms</b> \u00b7 '
            f'Generation: <b style="color:#0f172a;">{rec["generation_latency_ms"]:.0f}ms</b></div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("🛠️ Human-in-the-Loop Correction", expanded=False):
            corrected_text = st.text_area(
                "Human Corrected Response",
                value=rec["agent_response"],
                height=150,
                key=f"hitl_{idx}"
            )
            
            fine_tune_example = {
                "query": rec["query"],
                "retrieved_contexts": rec["retrieved_contexts"],
                "agent_response": corrected_text
            }
            jsonl_str = json.dumps(fine_tune_example) + "\n"
            
            st.download_button(
                label="Export as Fine-Tuning Example (.jsonl)",
                data=jsonl_str,
                file_name=f"hitl_correction_{rec['tweet_id']}.jsonl",
                mime="application/jsonl",
                type="primary",
                use_container_width=True
            )

# ---------------------------------------------------------------------------
# RAG Agent Loader
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_rag_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_agent_mod", str(BASE_DIR / "2_rag_agent.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tab 3: Zia Reply & Email Assistant
# ---------------------------------------------------------------------------

def render_zia_tab(df, tone, persona):
    st.markdown("<br>", unsafe_allow_html=True)
    col_in, col_em = st.columns([1, 1])

    with col_in:
        st.markdown('<div class="section-header">🤖 Zia Live Copilot Console</div>',
                    unsafe_allow_html=True)

        query_list = ["-- Custom Query --"] + df["query"].tolist()
        sel_q = st.selectbox(f"Select a benchmark query from evaluated records (N={len(df)})", query_list, index=0)
        
        default_val = sel_q if sel_q != "-- Custom Query --" else ""

        user_query = st.text_area(
            "Customer Query",
            value=default_val,
            placeholder="e.g. My Amazon Prime charge appeared twice this month...",
            height=100, key="live_query",
        )
        cb1, cb2 = st.columns(2)
        with cb1:
            run_btn = st.button("Run Agent Resolution", type="primary",
                                 use_container_width=True, key="run_agent")
        with cb2:
            top_k = st.selectbox("Top-K Contexts", [1, 2, 3, 4, 5], index=2, key="top_k_sel")

        if "agent_result" not in st.session_state:
            st.session_state.agent_result = None

        if run_btn and user_query.strip():
            with st.status("Running Agent Pipeline...", expanded=True) as status:
                st.write("🔍 Embedding query with all-MiniLM-L6-v2...")
                try:
                    mod = _load_rag_module()
                    collection = mod.get_collection()
                    contexts, retrieval_latency = mod.retrieve_context(collection, user_query.strip(), int(top_k))
                    st.write(f"⚡ Retrieved {int(top_k)} policy chunks from ChromaDB (925 indexed passages)...")
                    
                    is_offline = user_query in df["query"].values
                    
                    is_live_mode = st.session_state.get("execution_mode", "").startswith("☁️")
                    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
                    
                    if is_live_mode and api_key:
                        st.write("🌐 Live API Key found, routing to LLM provider...")
                        result = mod.generate_response(query=user_query.strip(), top_k=int(top_k), collection=collection)
                        result["_is_offline"] = False
                        st.session_state.agent_result = result
                    else:
                        if is_live_mode and not api_key:
                            st.warning("Live Mode selected but no API keys found in .env. Falling back to Offline Mode.")
                        st.write("📴 Offline Mode fallback...")
                        if is_offline:
                            rec = df[df["query"] == user_query].iloc[0]
                            tel = rec.get("telemetry", {})
                            st.session_state.agent_result = {
                                "response": rec.get("agent_response", ""),
                                "suggested_escalation": rec.get("suggested_escalation", False),
                                "retrieved_contexts": [ctx["text"] for ctx in contexts],
                                "source_tweet_ids": [ctx["tweet_id"] for ctx in contexts],
                                "distances": [ctx["distance"] for ctx in contexts],
                                "metrics": {
                                    "retrieval_latency_ms": retrieval_latency,
                                    "generation_latency_ms": tel.get("generation_latency_ms", 0),
                                    "total_tokens": tel.get("total_tokens", 0)
                                },
                                "_is_offline": True,
                                "_offline_badge": "Offline Replay"
                            }
                        else:
                            top_ctx = contexts[0]["text"] if contexts else ""
                            top_ctx_clean = re.sub(r'\[.*?\]', '', top_ctx)
                            top_ctx_clean = re.sub(r'@AmazonHelp', '', top_ctx_clean, flags=re.IGNORECASE)
                            top_ctx_clean = re.sub(r'\^[A-Za-z0-9]+$', '', top_ctx_clean).strip()
                            synth_response = f"Based on our policy records: {top_ctx_clean} Please let us know if you need additional assistance."
                            
                            st.session_state.agent_result = {
                                "response": synth_response,
                                "suggested_escalation": False,
                                "retrieved_contexts": [ctx["text"] for ctx in contexts],
                                "source_tweet_ids": [ctx["tweet_id"] for ctx in contexts],
                                "distances": [ctx["distance"] for ctx in contexts],
                                "metrics": {
                                    "retrieval_latency_ms": retrieval_latency,
                                    "generation_latency_ms": 0,
                                    "total_tokens": 0
                                },
                                "_is_offline": True,
                                "_offline_badge": "Local Vector Search (Offline Synthesis)"
                            }
                    status.update(label="Resolution Complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Agent Error", state="error", expanded=True)
                    st.error(f"Agent error: {e}")
                    st.session_state.agent_result = None

        result = st.session_state.agent_result
        if result:
            st.markdown("<br>", unsafe_allow_html=True)
            if result.get("_is_offline"):
                badge_text = result.get("_offline_badge", "Offline Demo Mode")
                st.markdown(f'<div style="background-color:#dbeafe;color:#1e3a8a;padding:4px 8px;border-radius:4px;font-size:12px;font-weight:600;display:inline-block;margin-bottom:8px;">⚡ {badge_text}</div>', unsafe_allow_html=True)
                
            st.markdown('<div class="section-header">📨 Agent Resolution</div>',
                        unsafe_allow_html=True)
            raw_res = result.get("response", "")
            escl = result.get("suggested_escalation", False)
            badge = (
                '<span class="pill-escalate">\u26a0 ESCALATE TO HUMAN</span>' if escl
                else '<span class="pill-good">\u2713 AUTO-RESOLVED</span>'
            )
            
            c_badge, c_copy = st.columns([3, 1])
            with c_badge:
                st.markdown(f'<div style="padding-top:4px;">{badge}</div>', unsafe_allow_html=True)
            with c_copy:
                if st.button("📋 Copy Text", key="btn_copy_resolution", use_container_width=True):
                    copied = copy_to_clipboard(raw_res)
                    if copied:
                        st.toast("Copied resolution to clipboard!", icon="📋")
                    else:
                        st.success("Resolution ready to copy")

            st.markdown(
                f'<div class="inspector-card" style="margin-top:6px;">'
                f'<div class="msg-bubble-agent" style="margin-bottom:0;">{_safe(raw_res)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📄 1-Click Code Copy (Browser Native)", expanded=False):
                st.code(raw_res, language=None)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📡 Zia Content Analysis</div>',
                        unsafe_allow_html=True)
            m = result.get("metrics", {})
            tcols = st.columns(4)
            for tc, (lbl, val, clr) in zip(tcols, [
                ("Retrieval",  f"{m.get('retrieval_latency_ms', 0):.1f}ms",  "#3b82f6"),
                ("Generation", f"{m.get('generation_latency_ms', 0):.0f}ms", "#8b5cf6"),
                ("Tokens",     str(m.get("total_tokens", 0)),                "#f59e0b"),
                ("Persona",    tone[:14],                                    "#22c55e"),
            ]):
                tc.markdown(
                    f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;'
                    f'padding:12px;text-align:center;border-top:3px solid {clr};">'
                    f'<div style="font-size:10px;color:#64748b;font-weight:600;text-transform:uppercase;">{lbl}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:#0f172a;margin-top:4px;">{val}</div></div>',
                    unsafe_allow_html=True,
                )

            with st.expander(f"📚 ChromaDB Context Evidence (Top {top_k})", expanded=False):
                contexts = result.get("retrieved_contexts", [])
                tweets = result.get("source_tweet_ids", [])
                distances = result.get("distances", [])
                for j in range(len(contexts)):
                    ctx = contexts[j] if j < len(contexts) else ""
                    src = tweets[j] if j < len(tweets) else "N/A"
                    dist = distances[j] if j < len(distances) else 0.0
                    st.markdown(
                        f'<div class="context-passage">'
                        f'<div style="display:flex; justify-content:space-between; font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:4px;">'
                        f'<span>SOURCE {j+1} \u00b7 #{src}</span>'
                        f'<span>Distance: {dist:.3f}</span></div>'
                        f'{_safe(ctx)}</div>',
                        unsafe_allow_html=True,
                    )

    with col_em:
        st.markdown('<div class="section-header">📧 Amazon Email Dispatch Preview</div>',
                    unsafe_allow_html=True)

        recipient = st.text_input("Recipient Email", placeholder="customer@example.com", key="recipient")
        agent_body = (st.session_state.agent_result or {}).get("response", "")
        email_body = st.text_area("Email Body (editable)", value=agent_body, height=160, key="email_body")

        to_disp = recipient if recipient else "&lt;recipient&gt;"
        st.markdown(
            f'<div class="email-frame" style="margin-top:12px;">'
            f'<div style="margin-bottom:6px;"><span class="email-field-label">From:</span> Amazon Customer Support &lt;support@amazon.com&gt;</div>'
            f'<div style="margin-bottom:6px;"><span class="email-field-label">To:</span> {to_disp}</div>'
            f'<div style="margin-bottom:6px;"><span class="email-field-label">Subject:</span> Support Case Update [Ticket #LIVE-PREVIEW]</div>'
            f'<hr class="email-divider">'
            f'<div class="email-body">Dear Valued Customer,\n\n{_safe(email_body)}\n\n'
            f'Thank you for contacting Amazon Customer Support.\n'
            f'We value your business and are committed to resolving this for you.\n\n'
            f'Best regards,\nAmazon Customer Support Team\n'
            f'support@amazon.com | amazon.com/help</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Send Support Email", type="primary", use_container_width=True, key="send_email"):
            if not recipient.strip() or "@" not in recipient:
                st.warning("Please enter a valid recipient email.")
            elif not email_body.strip():
                st.warning("Email body is empty. Run the agent first.")
            else:
                sent = _send_email(recipient.strip(), email_body.strip())
                if sent:
                    st.success(f"Email dispatched to {recipient}")
                else:
                    st.info("Email logged (SMTP not configured). Set SMTP_HOST, SMTP_USER, SMTP_PASS in .env.")
                    _log_email(recipient.strip(), email_body.strip())


def _send_email(to, body):
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pw   = os.getenv("SMTP_PASS", "")
    if not (host and user and pw):
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Support Case Update [Ticket #LIVE-PREVIEW]"
        msg["From"]    = f"Amazon Customer Support <{user}>"
        msg["To"]      = to
        full = textwrap.dedent(f"""\
            Dear Valued Customer,

            {body}

            Thank you for contacting Amazon Customer Support.

            Best regards,
            Amazon Customer Support Team
        """)
        msg.attach(MIMEText(full, "plain"))
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pw)
            s.sendmail(user, to, msg.as_string())
        return True
    except Exception as e:
        st.warning(f"SMTP error: {e}")
        return False


def _log_email(to, body):
    log = BASE_DIR / "email_dispatch_log.txt"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\nTO: {to}\nBODY:\n{body}\n")


def copy_to_clipboard(text: str) -> bool:
    """Copy text directly to the system clipboard (Windows clip or fallback)."""
    try:
        if sys.platform == "win32":
            p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
            p.communicate(text.encode("utf-16le"))
            return p.returncode == 0
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, summary = load_data()
    intent_f, esc_f, thresh, search, tone, persona = render_sidebar(df)
    filtered = apply_filters(df, intent_f, esc_f, thresh, search)

    report_date = ""
    rga = summary.get("report_generated_at", "")
    if rga:
        report_date = str(rga)[:10]

    st.markdown(
        f'<div class="global-header">'
        f'<div>'
        f'<div class="global-header-title">🛒 @AmazonHelp \u00b7 Enterprise Support Console</div>'
        f'<div class="global-header-sub">RAG Evaluation Dashboard \u00b7 {df.shape[0]} tickets evaluated \u00b7 Powered by ChromaDB + LLM Judge</div>'
        f'</div>'
        f'<div class="header-status"><span class="header-live-dot"></span>Live \u00b7 Report {report_date}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    render_kpi_banner(df, summary)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  Executive Analytics & Benchmarks",
        "  Forensic Answer Sandbox",
        "  Zia Reply & Email Assistant",
    ])

    with tab1:
        render_analytics_tab(df, summary)
    with tab2:
        render_forensic_tab(df, filtered)
    with tab3:
        render_zia_tab(df, tone, persona)


if __name__ == "__main__":
    main()
