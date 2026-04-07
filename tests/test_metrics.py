"""
Tests for evaluation metrics.
These are pure-Python unit tests with no external dependencies.
"""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    _extract_citation_ids,
    citation_accuracy,
    evaluate_response,
    faithfulness,
    hallucination_rate,
)


# ── _extract_citation_ids ─────────────────────────────────────────────────────

def test_extract_citation_ids_basic():
    text = "Revenue grew by 15% [1] while operating costs increased [2]."
    assert _extract_citation_ids(text) == [1, 2]


def test_extract_citation_ids_empty():
    assert _extract_citation_ids("No citations here.") == []


def test_extract_citation_ids_duplicates():
    text = "See [1] and [1] again, plus [3]."
    ids = _extract_citation_ids(text)
    assert ids == [1, 1, 3]


# ── citation_accuracy ─────────────────────────────────────────────────────────

def test_citation_accuracy_all_valid():
    report = "Revenue was $100M [1]. Operating income fell [2]."
    citations = [
        {"id": 1, "ticker": "AAPL", "form": "10-K", "filingDate": "2024-01-01", "accessionNumber": "0001", "chunk_index": 0},
        {"id": 2, "ticker": "AAPL", "form": "10-K", "filingDate": "2024-01-01", "accessionNumber": "0001", "chunk_index": 1},
    ]
    assert citation_accuracy(report, citations) == 1.0


def test_citation_accuracy_partial():
    report = "Revenue was $100M [1]. See [5] for details."  # [5] is invalid
    citations = [
        {"id": 1, "ticker": "AAPL", "form": "10-K", "filingDate": "2024-01-01", "accessionNumber": "0001", "chunk_index": 0},
    ]
    # 1 valid out of 2 referenced
    assert citation_accuracy(report, citations) == 0.5


def test_citation_accuracy_no_citations():
    report = "Revenue was $100M. Operating income fell."
    citations: list = []
    assert citation_accuracy(report, citations) == 1.0


def test_citation_accuracy_all_invalid():
    report = "Revenue [9] and costs [10] are important."
    citations = [{"id": 1, "ticker": "AAPL", "form": "10-K", "filingDate": "", "accessionNumber": "", "chunk_index": 0}]
    assert citation_accuracy(report, citations) == 0.0


# ── hallucination_rate ────────────────────────────────────────────────────────

def test_hallucination_rate_fully_grounded():
    context = [
        "Apple reported revenue of $394 billion in fiscal 2024 driven by iPhone sales.",
        "Operating income was $123 billion representing a significant increase year over year.",
    ]
    # Sentence uses words clearly from context
    report = "Apple reported revenue of $394 billion from iPhone sales in fiscal 2024."
    rate = hallucination_rate(report, context)
    # Should be low because most keywords match context
    assert rate <= 0.5


def test_hallucination_rate_fully_hallucinated():
    context = ["Apple reported strong revenue in fiscal 2024."]
    report = "Xenophobus launched a revolutionary quantum blockchain platform in 2099."
    rate = hallucination_rate(report, context)
    assert rate == 1.0


def test_hallucination_rate_empty_report():
    assert hallucination_rate("", ["some context"]) == 0.0


# ── faithfulness ─────────────────────────────────────────────────────────────

def test_faithfulness_complement_of_hallucination():
    context = ["Revenue increased significantly driven by cloud services."]
    report = "Revenue increased thanks to cloud services growth this year."
    hall = hallucination_rate(report, context)
    faith = faithfulness(report, context)
    assert abs(faith - (1.0 - hall)) < 1e-9


# ── evaluate_response ─────────────────────────────────────────────────────────

def test_evaluate_response_returns_all_keys():
    query = "What are Apple's revenue trends?"
    report = "Apple revenue grew [1]. Risks include competition [2]."
    citations = [
        {"id": 1, "ticker": "AAPL", "form": "10-K", "filingDate": "2024", "accessionNumber": "001", "chunk_index": 0},
        {"id": 2, "ticker": "AAPL", "form": "10-K", "filingDate": "2024", "accessionNumber": "001", "chunk_index": 1},
    ]
    docs_text = ["Apple revenue grew significantly in fiscal 2024.", "Competition risks include Samsung and Google."]

    result = evaluate_response(query, report, citations, docs_text)
    assert "citation_accuracy" in result
    assert "hallucination_rate" in result
    assert "faithfulness" in result
    assert "num_citations" in result
    assert "report_length_chars" in result
    assert result["num_citations"] == 2
    assert result["report_length_chars"] == len(report)
    # Scores must be in [0, 1]
    assert 0.0 <= result["citation_accuracy"] <= 1.0
    assert 0.0 <= result["hallucination_rate"] <= 1.0
    assert 0.0 <= result["faithfulness"] <= 1.0
