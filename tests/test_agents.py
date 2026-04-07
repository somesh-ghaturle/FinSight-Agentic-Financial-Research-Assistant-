"""
Tests for the LangGraph and CrewAI agents (mocked LLM and retriever).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agents.langgraph_agents import (
    ResearchState,
    _extract_citations,
    _format_docs,
    analyst_node,
    retriever_node,
    writer_node,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(content: str, ticker: str = "AAPL", form: str = "10-K", chunk: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={
            "ticker": ticker,
            "form": form,
            "filingDate": "2024-01-01",
            "accessionNumber": "0001193125-24-000001",
            "chunk_index": chunk,
        },
    )


def _base_state() -> ResearchState:
    return {
        "query": "What are Apple's revenue risks?",
        "retrieved_docs": [],
        "analysis": "",
        "final_report": "",
        "citations": [],
        "messages": [],
    }


# ── _format_docs ──────────────────────────────────────────────────────────────

def test_format_docs_includes_citation():
    docs = [_make_doc("Revenue grew by 10%.", ticker="AAPL")]
    formatted = _format_docs(docs)
    assert "[1]" in formatted
    assert "AAPL" in formatted
    assert "Revenue grew" in formatted


def test_format_docs_multiple():
    docs = [
        _make_doc("Revenue excerpt.", ticker="AAPL", chunk=0),
        _make_doc("Risk factors excerpt.", ticker="AAPL", chunk=1),
    ]
    formatted = _format_docs(docs)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "---" in formatted


# ── _extract_citations ────────────────────────────────────────────────────────

def test_extract_citations_structure():
    docs = [_make_doc("Test content", ticker="MSFT", form="10-Q")]
    citations = _extract_citations(docs)
    assert len(citations) == 1
    c = citations[0]
    assert c["id"] == 1
    assert c["ticker"] == "MSFT"
    assert c["form"] == "10-Q"
    assert c["filingDate"] == "2024-01-01"


# ── retriever_node ────────────────────────────────────────────────────────────

@patch("agents.langgraph_agents.get_retriever")
def test_retriever_node(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [_make_doc("Revenue section.")]
    mock_get_retriever.return_value = mock_retriever

    state = _base_state()
    new_state = retriever_node(state)

    assert len(new_state["retrieved_docs"]) == 1
    assert len(new_state["citations"]) == 1
    assert new_state["citations"][0]["ticker"] == "AAPL"
    mock_retriever.invoke.assert_called_once_with("What are Apple's revenue risks?")


@patch("agents.langgraph_agents.get_retriever")
def test_retriever_node_empty_results(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_get_retriever.return_value = mock_retriever

    state = _base_state()
    new_state = retriever_node(state)

    assert new_state["retrieved_docs"] == []
    assert new_state["citations"] == []


# ── analyst_node ──────────────────────────────────────────────────────────────

@patch("agents.langgraph_agents._llm")
def test_analyst_node(mock_llm_factory):
    mock_llm = MagicMock()
    mock_llm_factory.return_value = mock_llm
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Analysis: Revenue grew [1].")
    # Make prompt | llm return mock_chain
    mock_llm.__or__ = lambda self, other: mock_chain

    state = {
        **_base_state(),
        "retrieved_docs": [_make_doc("Revenue grew by 10%.")],
        "citations": [
            {"id": 1, "ticker": "AAPL", "form": "10-K", "filingDate": "2024", "accessionNumber": "001", "chunk_index": 0}
        ],
    }

    # Patch the ChatPromptTemplate chain construction
    with patch("agents.langgraph_agents.ChatPromptTemplate") as mock_prompt_cls:
        mock_prompt = MagicMock()
        mock_prompt_cls.from_messages.return_value = mock_prompt
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.invoke.return_value = MagicMock(content="Analysis: Revenue grew [1].")

        new_state = analyst_node(state)

    assert "analysis" in new_state
    assert "messages" in new_state


# ── writer_node ───────────────────────────────────────────────────────────────

@patch("agents.langgraph_agents.ChatPromptTemplate")
@patch("agents.langgraph_agents._llm")
def test_writer_node(mock_llm_factory, mock_prompt_cls):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="# Report\nApple revenue strong.")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)
    mock_prompt_cls.from_messages.return_value = mock_prompt

    state = {
        **_base_state(),
        "analysis": "Revenue grew significantly.",
        "retrieved_docs": [],
    }
    new_state = writer_node(state)
    assert "final_report" in new_state


# ── run_research_pipeline (integration, fully mocked) ─────────────────────────

@patch("agents.langgraph_agents.get_retriever")
@patch("agents.langgraph_agents.ChatPromptTemplate")
@patch("agents.langgraph_agents._llm")
def test_run_research_pipeline_keys(mock_llm_f, mock_prompt_cls, mock_retriever_f):
    from agents.langgraph_agents import run_research_pipeline

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [_make_doc("Some content.")]
    mock_retriever_f.return_value = mock_retriever

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Generated text.")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)
    mock_prompt_cls.from_messages.return_value = mock_prompt

    result = run_research_pipeline("Apple revenue risks")
    assert "query" in result
    assert "final_report" in result
    assert "citations" in result
