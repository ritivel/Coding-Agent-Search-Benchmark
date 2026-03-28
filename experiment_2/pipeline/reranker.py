"""Stage 5: Re-ranking — cross-encoder re-ranker for precision improvement."""

import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: dict


class CrossEncoderReranker:
    """Re-ranks retrieval results using a cross-encoder for higher precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        logger.info(f"Loading reranker model: {self.model_name}")
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)
        logger.info("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
        original_weight: float = 0.3,
    ) -> List[RankedResult]:
        if not results:
            return []

        self._load_model()

        pairs = [[query, r.text] for r in results]
        rerank_scores = self._model.predict(pairs)

        if len(rerank_scores) > 0:
            min_s = min(rerank_scores)
            max_s = max(rerank_scores)
            range_s = max_s - min_s if max_s != min_s else 1.0
            norm_scores = [(s - min_s) / range_s for s in rerank_scores]
        else:
            norm_scores = []

        ranked = []
        for i, (result, norm_score) in enumerate(zip(results, norm_scores)):
            final_score = (1 - original_weight) * norm_score + original_weight * result.score
            ranked.append(
                RankedResult(
                    text=result.text,
                    original_score=result.score,
                    rerank_score=float(rerank_scores[i]),
                    final_score=final_score,
                    metadata=result.metadata if hasattr(result, "metadata") else {},
                )
            )

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]
