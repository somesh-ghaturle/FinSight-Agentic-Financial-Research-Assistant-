"""
Tests for the RAG pipeline components (FAISS store, retriever factory).
External dependencies (OpenAI, FAISS) are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── faiss_store ───────────────────────────────────────────────────────────────

class TestFaissStore:
    @patch("rag.faiss_store.FAISS")
    @patch("rag.faiss_store.get_embeddings")
    def test_build_faiss_index_creates_docs(self, mock_embeddings, mock_faiss_cls, tmp_path):
        from rag.faiss_store import build_faiss_index

        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vectorstore

        filing = {"ticker": "AAPL", "form": "10-K", "filingDate": "2024-01-01",
                  "accessionNumber": "001", "text": "Revenue grew. " * 100}
        result = build_faiss_index([filing], persist_path=tmp_path / "faiss")

        assert mock_faiss_cls.from_documents.called
        mock_vectorstore.save_local.assert_called_once()

    @patch("rag.faiss_store.FAISS")
    @patch("rag.faiss_store.get_embeddings")
    def test_build_faiss_index_no_text_raises(self, mock_embeddings, mock_faiss_cls, tmp_path):
        from rag.faiss_store import build_faiss_index

        mock_embeddings.return_value = MagicMock()
        with pytest.raises(ValueError, match="No documents"):
            build_faiss_index([{"ticker": "AAPL", "text": ""}], persist_path=tmp_path / "faiss")

    def test_load_faiss_index_missing_raises(self, tmp_path):
        from rag.faiss_store import load_faiss_index

        with pytest.raises(FileNotFoundError):
            load_faiss_index(persist_path=tmp_path / "nonexistent")

    @patch("rag.faiss_store.FAISS")
    @patch("rag.faiss_store.get_embeddings")
    def test_load_faiss_index_success(self, mock_embeddings, mock_faiss_cls, tmp_path):
        from rag.faiss_store import load_faiss_index

        index_path = tmp_path / "faiss"
        index_path.mkdir()
        mock_embeddings.return_value = MagicMock()
        mock_faiss_cls.load_local.return_value = MagicMock()

        result = load_faiss_index(persist_path=index_path)
        mock_faiss_cls.load_local.assert_called_once()

    @patch("rag.faiss_store.load_faiss_index")
    def test_get_or_build_loads_existing(self, mock_load, tmp_path):
        from rag.faiss_store import get_or_build_faiss_index

        mock_store = MagicMock()
        mock_load.return_value = mock_store
        result = get_or_build_faiss_index()
        assert result is mock_store

    @patch("rag.faiss_store.build_faiss_index")
    @patch("rag.faiss_store.load_faiss_index")
    def test_get_or_build_builds_when_not_found(self, mock_load, mock_build, tmp_path):
        from rag.faiss_store import get_or_build_faiss_index

        mock_load.side_effect = FileNotFoundError("not found")
        mock_store = MagicMock()
        mock_build.return_value = mock_store

        filings = [{"ticker": "AAPL", "text": "Some text."}]
        result = get_or_build_faiss_index(filing_dicts=filings)
        assert result is mock_store
        mock_build.assert_called_once()


# ── retriever factory ─────────────────────────────────────────────────────────

class TestRetrieverFactory:
    @patch("rag.faiss_store.load_faiss_index")
    def test_get_retriever_faiss(self, mock_load):
        import config.settings as settings
        original = settings.VECTOR_STORE_BACKEND
        try:
            settings.VECTOR_STORE_BACKEND = "faiss"
            mock_store = MagicMock()
            mock_load.return_value = mock_store

            # Force reload so it picks up the new VECTOR_STORE_BACKEND
            import importlib
            import rag.retriever as retriever_mod
            importlib.reload(retriever_mod)
            retriever_mod.get_retriever()
            mock_store.as_retriever.assert_called_once()
        finally:
            settings.VECTOR_STORE_BACKEND = original

    @patch("rag.pinecone_store.get_pinecone_store")
    def test_get_retriever_pinecone(self, mock_pinecone):
        import config.settings as settings
        original = settings.VECTOR_STORE_BACKEND
        try:
            settings.VECTOR_STORE_BACKEND = "pinecone"
            mock_store = MagicMock()
            mock_pinecone.return_value = mock_store

            import importlib
            import rag.retriever as retriever_mod
            importlib.reload(retriever_mod)
            retriever_mod.get_retriever()
            mock_store.as_retriever.assert_called_once()
        finally:
            settings.VECTOR_STORE_BACKEND = original
