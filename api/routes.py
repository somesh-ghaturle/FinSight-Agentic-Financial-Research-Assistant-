"""
FastAPI route handlers for FinSight.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import (
    EvaluationSummaryResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from config.settings import LLM_MODEL, VECTOR_STORE_BACKEND

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return system health and configuration summary."""
    return HealthResponse(
        status="ok",
        vector_store_backend=VECTOR_STORE_BACKEND,
        llm_model=LLM_MODEL,
    )


# ── Research ──────────────────────────────────────────────────────────────────

@router.post("/research", response_model=QueryResponse, tags=["Research"])
def research(request: QueryRequest) -> QueryResponse:
    """
    Run the multi-agent research pipeline for a natural-language query.

    Choose *backend* = "langgraph" (default) or "crewai".
    Set *evaluate* = true to log quality metrics to MLflow.
    """
    try:
        if request.backend == "crewai":
            from agents.crewai_agents import run_crew_research

            result = run_crew_research(request.query)
            citations: List[dict] = []
        else:
            from agents.langgraph_agents import run_research_pipeline

            result = run_research_pipeline(request.query)
            citations = result.get("citations", [])

    except Exception as exc:  # noqa: BLE001
        logger.exception("Research pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    evaluation_result = None
    if request.evaluate:
        try:
            from rag.retriever import get_retriever
            from evaluation.mlflow_tracker import log_evaluation

            retriever = get_retriever()
            docs = retriever.invoke(request.query)
            docs_text = [d.page_content for d in docs]
            evaluation_result = log_evaluation(
                query=request.query,
                report=result["final_report"],
                citations=citations,
                retrieved_docs_text=docs_text,
                extra_params={"backend": request.backend},
                run_name=f"{request.backend}-query",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation logging failed: %s", exc)

    return QueryResponse(
        query=request.query,
        final_report=result["final_report"],
        analysis=result.get("analysis"),
        citations=citations,
        evaluation=evaluation_result,
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Data"])
def ingest_filings(request: IngestRequest) -> IngestResponse:
    """
    Fetch SEC filings for *ticker* and index them into the vector store.

    For FAISS: the index is persisted to disk.
    For Pinecone: documents are upserted to the cloud index.
    """
    try:
        from rag.sec_fetcher import fetch_filings_for_ticker

        filings = fetch_filings_for_ticker(
            ticker=request.ticker,
            form_types=request.form_types,
            max_filings=request.max_filings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Filing fetch error for %s", request.ticker)
        raise HTTPException(status_code=502, detail=f"SEC EDGAR error: {exc}") from exc

    if not filings:
        raise HTTPException(
            status_code=404,
            detail=f"No {request.form_types} filings found for {request.ticker}",
        )

    try:
        if VECTOR_STORE_BACKEND == "pinecone":
            from rag.pinecone_store import upsert_filings_to_pinecone

            upsert_filings_to_pinecone(filings)
        else:
            from rag.faiss_store import get_or_build_faiss_index

            get_or_build_faiss_index(filing_dicts=filings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store indexing error")
        raise HTTPException(status_code=500, detail=f"Indexing error: {exc}") from exc

    # Count indexed chunks (approximate based on CHUNK_SIZE)
    from config.settings import CHUNK_SIZE

    total_text_len = sum(len(f.get("text", "")) for f in filings)
    approx_chunks = max(1, total_text_len // CHUNK_SIZE)

    return IngestResponse(
        ticker=request.ticker.upper(),
        filings_ingested=len(filings),
        chunks_indexed=approx_chunks,
        message=f"Successfully indexed {len(filings)} filing(s) for {request.ticker.upper()}",
    )


# ── Evaluation summary ────────────────────────────────────────────────────────

@router.get(
    "/evaluation/summary",
    response_model=EvaluationSummaryResponse,
    tags=["Evaluation"],
)
def evaluation_summary() -> EvaluationSummaryResponse:
    """Return aggregate evaluation metrics from the MLflow experiment."""
    try:
        from evaluation.mlflow_tracker import get_experiment_summary

        summary = get_experiment_summary()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to retrieve MLflow summary")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EvaluationSummaryResponse(
        total_runs=summary.get("total_runs", 0),
        avg_citation_accuracy=summary.get("avg_citation_accuracy"),
        avg_hallucination_rate=summary.get("avg_hallucination_rate"),
        avg_faithfulness=summary.get("avg_faithfulness"),
    )
