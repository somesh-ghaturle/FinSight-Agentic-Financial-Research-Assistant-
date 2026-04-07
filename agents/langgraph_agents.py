"""
LangGraph multi-agent orchestration for FinSight.

Graph topology:
  START ──► retriever_node ──► analyst_node ──► writer_node ──► END

State is a typed dict that flows through the graph.
"""

from __future__ import annotations

import logging
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config.settings import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    query: str
    retrieved_docs: List[Document]
    analysis: str
    final_report: str
    citations: List[dict]
    messages: Annotated[List[BaseMessage], lambda a, b: a + b]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citation = (
            f"[{i}] {meta.get('ticker', 'N/A')} "
            f"{meta.get('form', '')} "
            f"({meta.get('filingDate', 'unknown date')})"
        )
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(docs: List[Document]) -> List[dict]:
    """Build a citation list from document metadata."""
    citations = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citations.append(
            {
                "id": i,
                "ticker": meta.get("ticker", "N/A"),
                "form": meta.get("form", ""),
                "filingDate": meta.get("filingDate", ""),
                "accessionNumber": meta.get("accessionNumber", ""),
                "chunk_index": meta.get("chunk_index", 0),
            }
        )
    return citations


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retriever_node(state: ResearchState) -> ResearchState:
    """Retrieve relevant SEC filing chunks for the user query."""
    logger.info("[LangGraph] retriever_node: query=%s", state["query"])
    retriever = get_retriever()
    docs = retriever.invoke(state["query"])
    citations = _extract_citations(docs)
    return {
        **state,
        "retrieved_docs": docs,
        "citations": citations,
        "messages": [HumanMessage(content=state["query"])],
    }


def analyst_node(state: ResearchState) -> ResearchState:
    """Analyse the retrieved documents and produce structured insights."""
    logger.info("[LangGraph] analyst_node")
    context = _format_docs(state["retrieved_docs"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert financial analyst specialising in SEC filings. "
                "Analyse the provided filing excerpts and produce structured insights "
                "covering: revenue trends, profitability, risk factors, and key metrics. "
                "Always reference the citation number [N] when citing specific data.",
            ),
            (
                "human",
                "Query: {query}\n\nFiling excerpts:\n{context}\n\n"
                "Provide a detailed financial analysis with citations.",
            ),
        ]
    )
    chain = prompt | _llm()
    analysis = chain.invoke({"query": state["query"], "context": context})
    return {
        **state,
        "analysis": analysis.content,
        "messages": [AIMessage(content=analysis.content)],
    }


def writer_node(state: ResearchState) -> ResearchState:
    """Synthesise the analysis into a well-structured research report."""
    logger.info("[LangGraph] writer_node")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial report writer. Convert the analysis into a concise, "
                "professional research report suitable for investors. "
                "Include an executive summary, key findings, risk assessment, and conclusion. "
                "Preserve all citation references [N].",
            ),
            (
                "human",
                "Analysis:\n{analysis}\n\nOriginal query: {query}\n\n"
                "Write the final investment research report.",
            ),
        ]
    )
    chain = prompt | _llm()
    report = chain.invoke({"analysis": state["analysis"], "query": state["query"]})
    return {
        **state,
        "final_report": report.content,
        "messages": [AIMessage(content=report.content)],
    }


# ── Graph construction ────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """Build and compile the FinSight LangGraph research pipeline."""
    graph = StateGraph(ResearchState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_research_pipeline(query: str) -> dict:
    """
    Execute the full LangGraph research pipeline for *query*.

    Returns a dict with keys: query, final_report, analysis, citations.
    """
    app = build_research_graph()
    initial_state: ResearchState = {
        "query": query,
        "retrieved_docs": [],
        "analysis": "",
        "final_report": "",
        "citations": [],
        "messages": [],
    }
    final_state = app.invoke(initial_state)
    return {
        "query": query,
        "final_report": final_state["final_report"],
        "analysis": final_state["analysis"],
        "citations": final_state["citations"],
    }
