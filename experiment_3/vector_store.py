"""
Vector store module using FAISS + sentence-transformers.

Handles embedding generation, FAISS index management, persistence, and
similarity search for the RAG retrieval step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

import config
from text_processing import TextChunk

logger = logging.getLogger(__name__)


class EmbeddingWrapper:
    """
    Wraps sentence-transformers so it satisfies the LangChain Embeddings
    interface (embed_documents / embed_query).
    """

    def __init__(self, model_name: str | None = None):
        name = model_name or config.EMBEDDING_MODEL
        logger.info("Loading embedding model: %s", name)
        self._model = SentenceTransformer(name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


class VectorStoreManager:
    """Manages a FAISS-backed vector store."""

    def __init__(self, embedder: EmbeddingWrapper | None = None):
        self.embedder = embedder or EmbeddingWrapper()
        self.store: FAISS | None = None

    def build_from_chunks(self, chunks: list[TextChunk]) -> FAISS:
        """Create a new FAISS index from a list of text chunks."""
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source_file,
                    "chunk_index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )
            for chunk in chunks
        ]

        logger.info("Building FAISS index from %d chunks...", len(documents))
        self.store = FAISS.from_documents(documents, self.embedder)
        logger.info("FAISS index ready (dimension=%d).", self.embedder.dimension)
        return self.store

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Add more chunks to an existing index."""
        if self.store is None:
            self.build_from_chunks(chunks)
            return

        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source_file,
                    "chunk_index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )
            for chunk in chunks
        ]
        self.store.add_documents(documents)

    def search(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Similarity search returning (document, score) pairs."""
        if self.store is None:
            raise RuntimeError("Vector store is empty — ingest documents first.")

        k = k or config.TOP_K_RESULTS
        results = self.store.similarity_search_with_score(query, k=k)

        threshold = score_threshold or config.SIMILARITY_THRESHOLD
        return [(doc, score) for doc, score in results if score <= threshold or True]

    def save(self, path: Path | None = None) -> None:
        """Persist index to disk."""
        if self.store is None:
            return
        save_path = str(path or config.VECTOR_STORE_DIR)
        self.store.save_local(save_path)
        logger.info("Vector store saved to %s", save_path)

    def load(self, path: Path | None = None) -> FAISS:
        """Load a persisted index."""
        load_path = str(path or config.VECTOR_STORE_DIR)
        self.store = FAISS.load_local(
            load_path,
            self.embedder,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vector store loaded from %s", load_path)
        return self.store
