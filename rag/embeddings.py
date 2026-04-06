"""
Embedding utilities used throughout FinSight.

Returns a LangChain-compatible embedding object based on the configured model.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from config.settings import EMBEDDING_MODEL, OPENAI_API_KEY


def get_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance configured from settings."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
