"""
Tests for the FastAPI routes using TestClient (no real LLM or vector store calls).
All external dependencies are mocked.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Pre-import modules so patch() can resolve dotted paths
import agents.langgraph_agents  # noqa: F401
import agents.crewai_agents  # noqa: F401
import rag.retriever  # noqa: F401
import rag.faiss_store  # noqa: F401
import rag.sec_fetcher  # noqa: F401
import evaluation.mlflow_tracker  # noqa: F401

from api.main import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_check():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vector_store_backend" in data
    assert "llm_model" in data


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "FinSight" in resp.json()["service"]


# ── Research ───────────────────────────────────────────────────────────────────

@patch("agents.langgraph_agents.run_research_pipeline")
@patch("rag.retriever.get_retriever")
@patch("evaluation.mlflow_tracker.log_evaluation")
def test_research_langgraph(mock_eval, mock_retriever, mock_pipeline):
    mock_doc = MagicMock()
    mock_doc.page_content = "Apple revenue grew."
    mock_retriever.return_value.invoke.return_value = [mock_doc]
    mock_pipeline.return_value = {
        "query": "Apple revenue",
        "final_report": "Apple had strong revenue [1].",
        "analysis": "Detailed analysis.",
        "citations": [
            {
                "id": 1,
                "ticker": "AAPL",
                "form": "10-K",
                "filingDate": "2024-01-01",
                "accessionNumber": "0001193125-24-000001",
                "chunk_index": 0,
            }
        ],
    }
    mock_eval.return_value = {
        "citation_accuracy": 1.0,
        "hallucination_rate": 0.1,
        "faithfulness": 0.9,
        "num_citations": 1,
        "report_length_chars": 30,
    }

    resp = client.post(
        "/api/v1/research",
        json={"query": "Apple revenue", "backend": "langgraph", "evaluate": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_report"] == "Apple had strong revenue [1]."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["ticker"] == "AAPL"
    assert data["evaluation"]["citation_accuracy"] == 1.0


@patch("agents.crewai_agents.run_crew_research")
def test_research_crewai(mock_crew):
    mock_crew.return_value = {
        "query": "MSFT risks",
        "final_report": "Microsoft faces competition risks.",
    }
    resp = client.post(
        "/api/v1/research",
        json={"query": "MSFT risks", "backend": "crewai", "evaluate": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Microsoft" in data["final_report"]
    assert data["citations"] == []


def test_research_invalid_backend():
    resp = client.post(
        "/api/v1/research",
        json={"query": "test", "backend": "invalid_backend", "evaluate": False},
    )
    assert resp.status_code == 422  # Pydantic validation error


def test_research_empty_query():
    resp = client.post(
        "/api/v1/research",
        json={"query": "ab", "backend": "langgraph", "evaluate": False},
    )
    assert resp.status_code == 422


# ── Ingest ─────────────────────────────────────────────────────────────────────

@patch("rag.sec_fetcher.fetch_filings_for_ticker")
@patch("rag.faiss_store.get_or_build_faiss_index")
def test_ingest_success(mock_index, mock_fetch):
    mock_fetch.return_value = [
        {
            "ticker": "AAPL",
            "form": "10-K",
            "filingDate": "2024-01-01",
            "accessionNumber": "0001193125-24-000001",
            "text": "A" * 5000,
        }
    ]
    mock_index.return_value = MagicMock()

    resp = client.post(
        "/api/v1/ingest",
        json={"ticker": "AAPL", "form_types": ["10-K"], "max_filings": 1},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["filings_ingested"] == 1
    assert "Successfully indexed" in data["message"]


@patch("rag.sec_fetcher.fetch_filings_for_ticker")
def test_ingest_no_filings_found(mock_fetch):
    mock_fetch.return_value = []
    resp = client.post(
        "/api/v1/ingest",
        json={"ticker": "XXXX", "form_types": ["10-K"], "max_filings": 1},
    )
    assert resp.status_code == 404


@patch("rag.sec_fetcher.fetch_filings_for_ticker")
def test_ingest_sec_error(mock_fetch):
    mock_fetch.side_effect = RuntimeError("EDGAR timeout")
    resp = client.post(
        "/api/v1/ingest",
        json={"ticker": "AAPL", "form_types": ["10-K"], "max_filings": 1},
    )
    assert resp.status_code == 502


# ── Evaluation summary ─────────────────────────────────────────────────────────

@patch("evaluation.mlflow_tracker.get_experiment_summary")
def test_evaluation_summary(mock_summary):
    mock_summary.return_value = {
        "total_runs": 10,
        "avg_citation_accuracy": 0.92,
        "avg_hallucination_rate": 0.08,
        "avg_faithfulness": 0.92,
    }
    resp = client.get("/api/v1/evaluation/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_runs"] == 10
    assert data["avg_citation_accuracy"] == 0.92
