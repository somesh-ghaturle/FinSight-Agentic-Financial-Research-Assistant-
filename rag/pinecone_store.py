"""
Pinecone-backed vector store for SEC filings.

Supports:
- Upserting chunked documents with metadata
- Similarity search with citation metadata
- Index creation if it doesn't exist
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PINECONE_API_KEY,
    PINECONE_DIMENSION,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
)
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _get_pinecone_client():
    """Return an initialised Pinecone client (v3+)."""
    from pinecone import Pinecone  # noqa: PLC0415

    return Pinecone(api_key=PINECONE_API_KEY)


def _ensure_index_exists(pc) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    from pinecone import ServerlessSpec  # noqa: PLC0415

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        logger.info("Created Pinecone index '%s'", PINECONE_INDEX_NAME)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def upsert_filings_to_pinecone(filing_dicts: List[dict]) -> PineconeVectorStore:
    """
    Chunk and upsert a list of filing dicts into Pinecone.

    Returns the PineconeVectorStore instance.
    """
    pc = _get_pinecone_client()
    _ensure_index_exists(pc)

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
        raise ValueError("No documents to upsert.")

    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    logger.info("Upserted %d chunks into Pinecone index '%s'", len(docs), PINECONE_INDEX_NAME)
    return vectorstore


def get_pinecone_store() -> PineconeVectorStore:
    """Return a PineconeVectorStore connected to the existing index."""
    pc = _get_pinecone_client()
    _ensure_index_exists(pc)
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
