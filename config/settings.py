"""
Central configuration and settings for FinSight.
All values are read from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Project paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Vector stores ─────────────────────────────────────────────────────────────
# Choose "faiss" (local, no key needed) or "pinecone" (cloud)
VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "faiss")

# Pinecone (only needed when VECTOR_STORE_BACKEND == "pinecone")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "finsight-sec-filings")
PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "1536"))

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
SEC_EDGAR_BASE_URL: str = "https://data.sec.gov"
SEC_EDGAR_USER_AGENT: str = os.getenv(
    "SEC_EDGAR_USER_AGENT",
    "FinSight research@finsight.example.com",
)
SEC_MAX_FILINGS: int = int(os.getenv("SEC_MAX_FILINGS", "5"))
# Chunk configuration for filing text splitting
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"

# ── Streamlit ─────────────────────────────────────────────────────────────────
STREAMLIT_API_URL: str = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "finsight-evaluation")

# ── CrewAI / LangGraph ────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "5"))
