"""
FinSight – Streamlit Frontend

A conversational UI for querying the FinSight multi-agent research assistant.
Communicates with the FastAPI backend.
"""

from __future__ import annotations

import time

import requests
import streamlit as st

from config.settings import STREAMLIT_API_URL

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight – Financial Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = STREAMLIT_API_URL.rstrip("/") + "/api/v1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_get(path: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _api_post(path: str, payload: dict, timeout: int = 120) -> dict:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return {"error": detail}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/320px-Amazon_logo.svg.png",
        width=80,
    )
    st.title("FinSight ⚙️")
    st.caption("Agentic Financial Research Assistant")

    st.divider()

    # Health check
    health = _api_get("/health")
    if "error" in health:
        st.error(f"API unreachable: {health['error']}")
    else:
        st.success("✅ API Connected")
        st.info(
            f"**LLM:** {health.get('llm_model', 'N/A')}\n\n"
            f"**Vector Store:** {health.get('vector_store_backend', 'N/A')}"
        )

    st.divider()

    # Ingest panel
    st.subheader("📥 Ingest SEC Filings")
    ticker_input = st.text_input("Ticker Symbol", placeholder="e.g. AAPL, MSFT")
    form_types = st.multiselect(
        "Form Types",
        ["10-K", "10-Q", "8-K"],
        default=["10-K"],
    )
    max_filings = st.slider("Max Filings", 1, 10, 3)

    if st.button("Ingest Filings", type="primary"):
        if not ticker_input.strip():
            st.warning("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Fetching {form_types} filings for {ticker_input.upper()}…"):
                result = _api_post(
                    "/ingest",
                    {
                        "ticker": ticker_input.strip().upper(),
                        "form_types": form_types,
                        "max_filings": max_filings,
                    },
                    timeout=180,
                )
            if "error" in result:
                st.error(f"Ingest failed: {result['error']}")
            else:
                st.success(result.get("message", "Ingested successfully!"))
                st.metric("Filings Ingested", result.get("filings_ingested", 0))
                st.metric("Chunks Indexed", result.get("chunks_indexed", 0))

    st.divider()

    # Evaluation summary
    st.subheader("📈 Evaluation Metrics")
    if st.button("Refresh Metrics"):
        st.session_state["eval_summary"] = _api_get("/evaluation/summary")

    if "eval_summary" in st.session_state:
        summary = st.session_state["eval_summary"]
        if "error" in summary:
            st.warning(f"Could not load metrics: {summary['error']}")
        else:
            st.metric("Total Runs", summary.get("total_runs", 0))
            if summary.get("avg_citation_accuracy") is not None:
                st.metric(
                    "Avg Citation Accuracy",
                    f"{summary['avg_citation_accuracy']:.2%}",
                )
            if summary.get("avg_hallucination_rate") is not None:
                st.metric(
                    "Avg Hallucination Rate",
                    f"{summary['avg_hallucination_rate']:.2%}",
                )
            if summary.get("avg_faithfulness") is not None:
                st.metric(
                    "Avg Faithfulness",
                    f"{summary['avg_faithfulness']:.2%}",
                )


# ── Main content ───────────────────────────────────────────────────────────────

st.title("📊 FinSight – Financial Research Assistant")
st.caption(
    "Ask questions about SEC filings. The multi-agent system retrieves relevant "
    "excerpts from 10-K, 10-Q, and 8-K filings and synthesises a cited research report."
)

# Agent backend selection
col1, col2 = st.columns([3, 1])
with col2:
    backend = st.selectbox(
        "Agent Backend",
        ["langgraph", "crewai"],
        index=0,
        help="LangGraph (graph-based) or CrewAI (role-based) agent backend",
    )
    run_eval = st.checkbox("Log to MLflow", value=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Citations"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**[{c['id']}]** {c['ticker']} · {c['form']} · {c['filingDate']} "
                        f"· chunk #{c['chunk_index']}"
                    )
        if msg.get("evaluation"):
            ev = msg["evaluation"]
            with st.expander("📊 Evaluation Metrics"):
                cols = st.columns(3)
                cols[0].metric("Citation Accuracy", f"{ev.get('citation_accuracy', 0):.2%}")
                cols[1].metric("Hallucination Rate", f"{ev.get('hallucination_rate', 0):.2%}")
                cols[2].metric("Faithfulness", f"{ev.get('faithfulness', 0):.2%}")

# Chat input
if query := st.chat_input("Ask about SEC filings… (e.g. 'What are Apple's main revenue risks in its latest 10-K?')"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            t0 = time.time()
            result = _api_post(
                "/research",
                {"query": query, "backend": backend, "evaluate": run_eval},
                timeout=300,
            )
            elapsed = time.time() - t0

        if "error" in result:
            st.error(f"Research failed: {result['error']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ Error: {result['error']}"}
            )
        else:
            report = result.get("final_report", "")
            citations = result.get("citations", [])
            evaluation = result.get("evaluation")

            st.markdown(report)
            st.caption(f"_Completed in {elapsed:.1f}s using {backend}_")

            if citations:
                with st.expander(f"📚 Citations ({len(citations)})"):
                    for c in citations:
                        st.markdown(
                            f"**[{c['id']}]** {c['ticker']} · "
                            f"{c['form']} · {c['filingDate']} · "
                            f"chunk #{c['chunk_index']}  \n"
                            f"Accession: `{c['accessionNumber']}`"
                        )

            if evaluation:
                with st.expander("📊 Evaluation Metrics"):
                    ev_cols = st.columns(3)
                    ev_cols[0].metric(
                        "Citation Accuracy",
                        f"{evaluation.get('citation_accuracy', 0):.2%}",
                    )
                    ev_cols[1].metric(
                        "Hallucination Rate",
                        f"{evaluation.get('hallucination_rate', 0):.2%}",
                    )
                    ev_cols[2].metric(
                        "Faithfulness",
                        f"{evaluation.get('faithfulness', 0):.2%}",
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": report,
                    "citations": citations,
                    "evaluation": evaluation,
                }
            )
