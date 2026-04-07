"""
Pydantic request/response models for the FinSight FastAPI backend.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /research endpoint."""

    query: str = Field(..., min_length=3, description="Natural-language research query")
    backend: str = Field(
        "langgraph",
        pattern="^(langgraph|crewai)$",
        description="Agent backend to use: 'langgraph' or 'crewai'",
    )
    evaluate: bool = Field(
        True,
        description="Whether to run the evaluation pipeline and log metrics to MLflow",
    )


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL, MSFT)")
    form_types: List[str] = Field(
        default=["10-K"],
        description="SEC form types to ingest (10-K, 10-Q, 8-K)",
    )
    max_filings: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of filings to ingest",
    )


# ── Response models ────────────────────────────────────────────────────────────

class CitationModel(BaseModel):
    id: int
    ticker: str
    form: str
    filingDate: str
    accessionNumber: str
    chunk_index: int


class QueryResponse(BaseModel):
    query: str
    final_report: str
    analysis: Optional[str] = None
    citations: List[CitationModel] = []
    evaluation: Optional[Dict] = None


class IngestResponse(BaseModel):
    ticker: str
    filings_ingested: int
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store_backend: str
    llm_model: str


class EvaluationSummaryResponse(BaseModel):
    total_runs: int
    avg_citation_accuracy: Optional[float] = None
    avg_hallucination_rate: Optional[float] = None
    avg_faithfulness: Optional[float] = None
