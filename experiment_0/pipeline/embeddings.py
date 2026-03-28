"""
Embedding engine using sentence-transformers for semantic search.

Converts text chunks into dense vector representations that power
the retrieval component of the RAG pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Produce dense embeddings using a sentence-transformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded — dimension=%d", self.dimension)

    @property
    def dimension(self) -> int:
        self._load()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._load()
        assert self._model is not None
        embeddings = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        logger.info("Embedded %d texts → shape %s", len(texts), embeddings.shape)
        return embeddings  # type: ignore[return-value]

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
