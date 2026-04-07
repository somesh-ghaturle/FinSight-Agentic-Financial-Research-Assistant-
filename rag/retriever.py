"""
RAG retriever factory.

Returns a LangChain retriever backed by either FAISS or Pinecone,
depending on the VECTOR_STORE_BACKEND setting.
"""

from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import RETRIEVER_TOP_K, VECTOR_STORE_BACKEND


def get_retriever() -> VectorStoreRetriever:
    """Return the configured vector-store retriever."""
    if VECTOR_STORE_BACKEND == "pinecone":
        from rag.pinecone_store import get_pinecone_store

        store = get_pinecone_store()
    else:
        from rag.faiss_store import load_faiss_index

        store = load_faiss_index()

    return store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
