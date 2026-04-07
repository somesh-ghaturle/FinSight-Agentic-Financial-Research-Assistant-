"""
FAISS-backed vector store for SEC filings.

Supports:
- Building an index from raw text chunks
- Persisting / loading from disk
- Similarity search with metadata (citation support)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, FAISS_INDEX_DIR
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def build_faiss_index(
    filing_dicts: List[dict],
    persist_path: Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Build a FAISS index from a list of filing dicts.

    Each dict must have at least ``text`` and any metadata keys
    (ticker, form, filingDate, accessionNumber).
    """
    splitter = _splitter()
    docs: List[Document] = []
    for filing in filing_dicts:
        raw_text = filing.get("text", "")
        if not raw_text:
            continue
        metadata = {k: v for k, v in filing.items() if k != "text"}
        chunks = splitter.split_text(raw_text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk_index": i},
                )
            )

    if not docs:
        raise ValueError("No documents to index. Provide at least one filing with non-empty text.")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    persist_path = Path(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))
    logger.info("FAISS index saved to %s (%d chunks)", persist_path, len(docs))
    return vectorstore


def load_faiss_index(persist_path: Path = FAISS_INDEX_DIR) -> FAISS:
    """Load an existing FAISS index from disk."""
    persist_path = Path(persist_path)
    if not persist_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {persist_path}. Run ingest first.")
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from %s", persist_path)
    return vectorstore


def get_or_build_faiss_index(
    filing_dicts: List[dict] | None = None,
    persist_path: Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Return the FAISS index, loading from disk if available, otherwise building it.

    If *filing_dicts* is provided and the index does not yet exist it will be
    built and persisted automatically.
    """
    try:
        return load_faiss_index(persist_path)
    except FileNotFoundError:
        if not filing_dicts:
            raise
        return build_faiss_index(filing_dicts, persist_path)
