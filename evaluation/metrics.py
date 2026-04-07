"""
Evaluation metrics for FinSight agent outputs.

Metrics:
- citation_accuracy   : fraction of citations that are traceable to retrieved docs
- hallucination_rate  : fraction of factual claims NOT supported by retrieved docs
- faithfulness        : RAGAS-style faithfulness score (claims supported / total claims)
"""

from __future__ import annotations

import re
from typing import List


def _extract_citation_ids(text: str) -> List[int]:
    """Extract all [N] citation references from *text*."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]


def citation_accuracy(report: str, citations: List[dict]) -> float:
    """
    Measure the fraction of cited IDs in *report* that exist in *citations*.

    A citation is considered accurate if its ID appears in the citations list.

    Returns a float in [0, 1].  Returns 1.0 if no citations are referenced.
    """
    referenced_ids = set(_extract_citation_ids(report))
    if not referenced_ids:
        return 1.0

    valid_ids = {c["id"] for c in citations}
    accurate = referenced_ids & valid_ids
    return len(accurate) / len(referenced_ids)


def _split_sentences(text: str) -> List[str]:
    """Split *text* into non-empty sentences using simple punctuation rules."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def hallucination_rate(report: str, context_docs: List[str]) -> float:
    """
    Estimate the hallucination rate as the fraction of sentences in *report*
    that cannot be loosely attributed to any passage in *context_docs*.

    Heuristic: a sentence is "grounded" if at least one context document
    contains ≥2 key content words from the sentence (ignoring stop words).

    Returns a float in [0, 1].  Returns 0.0 if no sentences are found.
    """
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "its", "it", "this", "that", "by", "as",
        "from", "we", "our", "their", "they", "he", "she", "which",
    }

    sentences = _split_sentences(report)
    if not sentences:
        return 0.0

    combined_context = " ".join(context_docs).lower()
    hallucinated = 0
    for sentence in sentences:
        words = {
            w.lower().strip(".,;:()[]")
            for w in sentence.split()
            if len(w) > 3 and w.lower() not in STOP_WORDS
        }
        # A sentence is grounded if ≥2 of its keywords appear in any context doc
        grounded = sum(1 for w in words if w in combined_context) >= 2
        if not grounded:
            hallucinated += 1

    return hallucinated / len(sentences)


def faithfulness(report: str, context_docs: List[str]) -> float:
    """
    Compute a faithfulness score: fraction of report sentences that are
    supported by at least one context passage.

    Faithfulness = 1 - hallucination_rate.

    Returns a float in [0, 1].
    """
    return 1.0 - hallucination_rate(report, context_docs)


def evaluate_response(
    query: str,
    report: str,
    citations: List[dict],
    retrieved_docs_text: List[str],
) -> dict:
    """
    Run all evaluation metrics and return a consolidated results dict.

    Parameters
    ----------
    query:               The original user query.
    report:              The generated research report text.
    citations:           The citation list from the research pipeline.
    retrieved_docs_text: Raw text content of retrieved documents.
    """
    cit_acc = citation_accuracy(report, citations)
    hall_rate = hallucination_rate(report, retrieved_docs_text)
    faith = faithfulness(report, retrieved_docs_text)

    return {
        "query": query,
        "citation_accuracy": round(cit_acc, 4),
        "hallucination_rate": round(hall_rate, 4),
        "faithfulness": round(faith, 4),
        "num_citations": len(citations),
        "report_length_chars": len(report),
    }
