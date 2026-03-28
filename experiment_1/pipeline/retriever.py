"""
Hybrid retriever combining BM25 (keyword) and vector (semantic) search
with Reciprocal Rank Fusion, followed by cross-encoder re-ranking.

This is the primary differentiator of Experiment 1 over Experiment 0:
  - BM25 catches exact keyword matches that embedding models miss
  - Vector search captures semantic similarity for paraphrased queries
  - RRF merges both ranked lists without needing score normalisation
  - Cross-encoder re-ranking prunes false positives with pairwise scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .chunker import Chunk
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with provenance."""

    chunk: Chunk
    score: float
    method: str  # "vector", "bm25", "hybrid", "reranked"


class HybridRetriever:
    """Two-stage retriever: hybrid search → cross-encoder re-ranking."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 0.7,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.reranker_model_name = reranker_model
        self._reranker: CrossEncoder | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []

    def _load_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info("Loading re-ranker: %s", self.reranker_model_name)
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def _build_bm25_index(self) -> None:
        """Build/rebuild BM25 index from all documents in the vector store."""
        docs = self.vector_store.get_all_documents()
        if not docs:
            logger.warning("No documents for BM25 index")
            self._bm25 = None
            self._bm25_corpus = []
            return
        self._bm25_corpus = docs
        tokenized = [doc.lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 index over %d documents", len(docs))

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (doc_index, bm25_score) pairs."""
        if self._bm25 is None:
            self._build_bm25_index()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(idx, float(scores[idx])) for idx in ranked_indices[:top_k]]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = {}

        for rank, (doc_id, _score) in enumerate(vector_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.alpha / (self.rrf_k + rank + 1)

        for rank, (doc_id, _score) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1 - self.alpha) / (self.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_candidates: int = 20,
        use_reranking: bool = True,
    ) -> list[RetrievalResult]:
        """Full hybrid search → re-rank pipeline.

        1. Vector search (top rerank_candidates)
        2. BM25 search (top rerank_candidates)
        3. RRF fusion
        4. Cross-encoder re-ranking on fused candidates → top_k
        """
        # Vector search
        vector_hits = self.vector_store.search(query, top_k=rerank_candidates)
        vector_ranked: list[tuple[str, float]] = []
        doc_map: dict[str, Chunk] = {}
        for i, (chunk, score) in enumerate(vector_hits):
            doc_id = f"v_{i}"
            vector_ranked.append((doc_id, score))
            doc_map[doc_id] = chunk

        # BM25 search
        self._build_bm25_index()
        bm25_hits = self._bm25_search(query, top_k=rerank_candidates)
        bm25_ranked: list[tuple[str, float]] = []
        for idx, score in bm25_hits:
            doc_id = f"b_{idx}"
            bm25_ranked.append((doc_id, score))
            if doc_id not in doc_map and idx < len(self._bm25_corpus):
                doc_map[doc_id] = Chunk(text=self._bm25_corpus[idx], source="bm25")

        # RRF fusion
        fused = self._reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        fused_candidates = [(doc_map[doc_id], score) for doc_id, score in fused if doc_id in doc_map]

        if not fused_candidates:
            logger.warning("No candidates after fusion")
            return []

        # Re-ranking
        if use_reranking and len(fused_candidates) > 1:
            reranker = self._load_reranker()
            pairs = [(query, chunk.text) for chunk, _score in fused_candidates]
            rerank_scores = reranker.predict(pairs)

            reranked = sorted(
                zip(fused_candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            results = [
                RetrievalResult(chunk=chunk, score=float(rscore), method="reranked")
                for (chunk, _fscore), rscore in reranked[:top_k]
            ]
        else:
            results = [
                RetrievalResult(chunk=chunk, score=score, method="hybrid")
                for chunk, score in fused_candidates[:top_k]
            ]

        logger.info(
            "Hybrid retrieval: %d vector + %d bm25 → %d fused → %d returned",
            len(vector_hits), len(bm25_hits), len(fused_candidates), len(results),
        )
        return results

    def vector_only_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Bypass hybrid/reranking for simple vector search."""
        hits = self.vector_store.search(query, top_k=top_k)
        return [
            RetrievalResult(chunk=chunk, score=score, method="vector")
            for chunk, score in hits
        ]
