"""
ChromaDB-based vector store for the RAG retrieval step.

Key difference from Experiment 0 (FAISS):
  - Persistent storage handled by ChromaDB natively
  - Built-in metadata filtering support
  - Collection-based organisation
  - No manual serialisation needed
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import numpy as np

from .chunker import Chunk
from .embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB persistent vector store with metadata support."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        persist_dir: Path | None = None,
        collection_name: str = "handwritten_docs",
    ):
        self.engine = embedding_engine
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def _ensure_collection(self) -> chromadb.Collection:
        if self._collection is not None:
            return self._collection

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d existing documents)",
            self.collection_name, self._collection.count(),
        )
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and add chunks to the ChromaDB collection."""
        collection = self._ensure_collection()

        texts = [c.text for c in chunks]
        embeddings = self.engine.embed_texts(texts)

        existing_count = collection.count()
        ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]
        metadatas = [c.to_metadata() for c in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        self._chunks.extend(chunks)
        logger.info("Added %d chunks to ChromaDB (total=%d)", len(chunks), collection.count())

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Semantic vector search, returning (Chunk, score) pairs."""
        collection = self._ensure_collection()
        if collection.count() == 0:
            logger.warning("Search on empty collection — returning nothing")
            return []

        query_emb = self.engine.embed_query(query)
        n_results = min(top_k, collection.count())

        query_params: dict = {
            "query_embeddings": [query_emb.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = collection.query(**query_params)

        output: list[tuple[Chunk, float]] = []
        if results["documents"] and results["distances"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],  # type: ignore[index]
                results["distances"][0],
            ):
                score = 1.0 - dist  # ChromaDB returns cosine distance; convert to similarity
                chunk = Chunk(
                    text=doc,
                    source=meta.get("source", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    ocr_confidence=meta.get("ocr_confidence", 0.0),
                )
                output.append((chunk, score))

        return output

    def get_all_documents(self) -> list[str]:
        """Return all stored document texts (for BM25 index building)."""
        collection = self._ensure_collection()
        if collection.count() == 0:
            return []
        result = collection.get(include=["documents"])
        return result["documents"] or []

    def get_all_ids(self) -> list[str]:
        """Return all document IDs in the collection."""
        collection = self._ensure_collection()
        if collection.count() == 0:
            return []
        result = collection.get()
        return result["ids"]

    def count(self) -> int:
        return self._ensure_collection().count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            self._chunks = []
            logger.info("Reset collection '%s'", self.collection_name)
