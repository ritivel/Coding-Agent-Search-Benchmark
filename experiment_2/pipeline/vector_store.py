"""Stage 4b: Vector Store — ChromaDB with metadata filtering and hybrid search support."""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0


class ChromaVectorStore:
    """ChromaDB-backed vector store with metadata filtering."""

    def __init__(
        self,
        collection_name: str = "handwritten_docs",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _init_store(self):
        if self._collection is not None:
            return
        import chromadb

        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready "
            f"({self._collection.count()} existing documents)"
        )

    def add_chunks(self, chunks) -> int:
        self._init_store()

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        existing_count = self._collection.count()

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{existing_count + i}"
            ids.append(chunk_id)
            embeddings.append(chunk.embedding.tolist() if isinstance(chunk.embedding, np.ndarray) else chunk.embedding)
            documents.append(chunk.text)

            meta = {
                "source_document": chunk.source_document,
                "chunk_index": chunk.chunk_index,
                "word_count": len(chunk.text.split()),
            }
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
            metadatas.append(meta)

        if ids:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(ids)} chunks to ChromaDB")

        return len(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        self._init_store()

        query_emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": min(top_k, max(self._collection.count(), 1)),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        retrieval_results = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)

            for doc, dist, meta in zip(docs, distances, metadatas):
                score = 1.0 - dist
                retrieval_results.append(
                    RetrievalResult(
                        text=doc,
                        score=score,
                        metadata=meta,
                        chunk_index=meta.get("chunk_index", 0),
                    )
                )

        return retrieval_results

    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        self._init_store()

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, max(self._collection.count(), 1)),
            )
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

        retrieval_results = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)

            for doc, dist, meta in zip(docs, distances, metadatas):
                retrieval_results.append(
                    RetrievalResult(
                        text=doc,
                        score=1.0 - dist,
                        metadata=meta,
                        chunk_index=meta.get("chunk_index", 0),
                    )
                )

        return retrieval_results

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> List[RetrievalResult]:
        vector_results = self.search(query_embedding, top_k=top_k * 2)
        keyword_results = self.keyword_search(query_text, top_k=top_k * 2)

        combined = {}
        for r in vector_results:
            combined[r.text] = RetrievalResult(
                text=r.text,
                score=r.score * vector_weight,
                metadata=r.metadata,
                chunk_index=r.chunk_index,
            )

        keyword_weight = 1.0 - vector_weight
        for r in keyword_results:
            if r.text in combined:
                combined[r.text].score += r.score * keyword_weight
            else:
                combined[r.text] = RetrievalResult(
                    text=r.text,
                    score=r.score * keyword_weight,
                    metadata=r.metadata,
                    chunk_index=r.chunk_index,
                )

        results = sorted(combined.values(), key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        self._init_store()
        return self._collection.count()

    def reset(self):
        self._init_store()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Reset collection '{self.collection_name}'")
