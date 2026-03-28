"""
Lightweight FAISS-based vector store for the RAG retrieval step.

Stores text chunks alongside their embeddings and supports
approximate nearest-neighbour search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from .embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    source: str = ""
    page: int = 0
    line_start: int = 0
    line_end: int = 0
    chunk_id: int = 0


class VectorStore:
    """FAISS flat-IP index with metadata bookkeeping."""

    def __init__(self, embedding_engine: EmbeddingEngine, persist_dir: Path | None = None):
        self.engine = embedding_engine
        self.persist_dir = persist_dir
        self.chunks: list[Chunk] = []
        self._index: faiss.IndexFlatIP | None = None

    def _ensure_index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            dim = self.engine.dimension
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Created FAISS IndexFlatIP with dim=%d", dim)
        return self._index

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
        source: str = "",
    ) -> list[Chunk]:
        words = text.split()
        chunks: list[Chunk] = []
        start = 0
        chunk_id = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(text=chunk_text, source=source, chunk_id=chunk_id)
            )
            chunk_id += 1
            start += chunk_size - overlap
        logger.info("Split text into %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
        return chunks

    def add_chunks(self, chunks: list[Chunk]) -> None:
        index = self._ensure_index()
        texts = [c.text for c in chunks]
        embeddings = self.engine.embed_texts(texts)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info("Added %d chunks to index (total=%d)", len(chunks), index.ntotal)

    def add_text(
        self,
        text: str,
        source: str = "",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[Chunk]:
        chunks = self.chunk_text(text, chunk_size, overlap, source)
        self.add_chunks(chunks)
        return chunks

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        index = self._ensure_index()
        if index.ntotal == 0:
            logger.warning("Search on empty index — returning nothing")
            return []

        query_vec = self.engine.embed_query(query).reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, min(top_k, index.ntotal))

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Path | None = None) -> None:
        directory = directory or self.persist_dir
        if directory is None:
            raise ValueError("No persist directory specified")
        directory.mkdir(parents=True, exist_ok=True)

        index = self._ensure_index()
        faiss.write_index(index, str(directory / "index.faiss"))

        meta = [
            {"text": c.text, "source": c.source, "page": c.page, "chunk_id": c.chunk_id}
            for c in self.chunks
        ]
        (directory / "chunks.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved vector store to %s", directory)

    def load(self, directory: Path | None = None) -> None:
        directory = directory or self.persist_dir
        if directory is None:
            raise ValueError("No persist directory specified")

        self._index = faiss.read_index(str(directory / "index.faiss"))
        meta = json.loads((directory / "chunks.json").read_text())
        self.chunks = [Chunk(**m) for m in meta]
        logger.info("Loaded vector store: %d chunks from %s", len(self.chunks), directory)
