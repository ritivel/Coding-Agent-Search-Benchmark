"""
TRACe evaluation framework — uTilization, Relevance, Adherence, Completeness.

Implements both:
  1. Ground-truth computation from RAGBench annotated spans/labels.
  2. Lightweight heuristic estimation when ground-truth annotations are absent
     (useful for evaluating new RAG model outputs at inference time).

Reference: "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation
Systems" — Friel, Belyi & Sanyal (2024). arXiv:2407.11005
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Data class for per-example TRACe scores
# ---------------------------------------------------------------------------

@dataclass
class TRACeScores:
    relevance: Optional[float] = None
    utilization: Optional[float] = None
    completeness: Optional[float] = None
    adherence: Optional[bool] = None


# ---------------------------------------------------------------------------
# Sentence tokeniser (lightweight, no NLTK dependency at import)
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using a simple regex fallback."""
    if not text:
        return []
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def _sentence_overlap(sentences_a: list[str], sentences_b: list[str]) -> list[str]:
    """Return sentences from *a* that have a near-duplicate in *b*."""
    norm_b = {_normalise(s) for s in sentences_b}
    return [s for s in sentences_a if _normalise(s) in norm_b]


def _fuzzy_sentence_contained(sentence: str, reference_text: str, threshold: float = 0.6) -> bool:
    """Check if a sentence is roughly contained in a reference text via token overlap."""
    sent_tokens = set(_normalise(sentence).split())
    ref_tokens = set(_normalise(reference_text).split())
    if not sent_tokens:
        return False
    overlap = len(sent_tokens & ref_tokens) / len(sent_tokens)
    return overlap >= threshold


# ---------------------------------------------------------------------------
# TRACe Evaluator
# ---------------------------------------------------------------------------

class TRACeEvaluator:
    """Compute TRACe metrics from example dicts."""

    def __init__(self, fuzzy_threshold: float = 0.6):
        self.fuzzy_threshold = fuzzy_threshold

    # ---- ground-truth path (labels already in the example) ----

    def scores_from_ground_truth(self, example: dict) -> TRACeScores:
        """Read pre-annotated TRACe labels directly from the example dict."""
        return TRACeScores(
            relevance=example.get("relevance"),
            utilization=example.get("utilization"),
            completeness=example.get("completeness"),
            adherence=example.get("adherence"),
        )

    # ---- heuristic estimation path (no labels, just text) ----

    def estimate_relevance(self, question: str, documents: list[str]) -> float:
        """
        Estimate Context Relevance = fraction of document sentences
        containing question-relevant information.
        Uses token overlap between each doc sentence and the question.
        """
        q_tokens = set(_normalise(question).split())
        if not q_tokens or not documents:
            return 0.0

        total_sents = 0
        relevant_sents = 0
        for doc in documents:
            for sent in _sent_tokenize(doc):
                total_sents += 1
                s_tokens = set(_normalise(sent).split())
                overlap = len(q_tokens & s_tokens) / len(q_tokens) if q_tokens else 0
                if overlap >= 0.2:
                    relevant_sents += 1

        return relevant_sents / total_sents if total_sents else 0.0

    def estimate_utilization(self, response: str, documents: list[str]) -> float:
        """
        Estimate Context Utilization = fraction of document sentences
        that are reflected in the response.
        """
        if not documents or not response:
            return 0.0

        total_sents = 0
        utilized_sents = 0
        for doc in documents:
            for sent in _sent_tokenize(doc):
                total_sents += 1
                if _fuzzy_sentence_contained(sent, response, self.fuzzy_threshold):
                    utilized_sents += 1

        return utilized_sents / total_sents if total_sents else 0.0

    def estimate_completeness(
        self, question: str, response: str, documents: list[str]
    ) -> float:
        """
        Completeness = Len(R ∩ U) / Len(R)
        i.e. fraction of *relevant* doc sentences that are also *utilized*.
        """
        q_tokens = set(_normalise(question).split())
        if not q_tokens or not documents:
            return 0.0

        relevant_sents: list[str] = []
        utilized_relevant = 0

        for doc in documents:
            for sent in _sent_tokenize(doc):
                s_tokens = set(_normalise(sent).split())
                is_relevant = (len(q_tokens & s_tokens) / len(q_tokens) >= 0.2) if q_tokens else False
                if is_relevant:
                    relevant_sents.append(sent)
                    if _fuzzy_sentence_contained(sent, response, self.fuzzy_threshold):
                        utilized_relevant += 1

        return utilized_relevant / len(relevant_sents) if relevant_sents else 0.0

    def estimate_adherence(self, response: str, documents: list[str]) -> bool:
        """
        Adherence (binary) = True iff every sentence in the response is
        grounded in at least one document.
        """
        if not response:
            return True
        combined_context = " ".join(documents)
        for sent in _sent_tokenize(response):
            if not _fuzzy_sentence_contained(sent, combined_context, self.fuzzy_threshold):
                return False
        return True

    def estimate_scores(
        self, question: str, response: str, documents: list[str]
    ) -> TRACeScores:
        """Compute all four TRACe metrics heuristically from raw text."""
        return TRACeScores(
            relevance=self.estimate_relevance(question, documents),
            utilization=self.estimate_utilization(response, documents),
            completeness=self.estimate_completeness(question, response, documents),
            adherence=self.estimate_adherence(response, documents),
        )

    # ---- batch helpers ----

    def evaluate_batch(
        self, examples: list[dict], use_ground_truth: bool = True
    ) -> list[TRACeScores]:
        """Evaluate a batch of examples, returning per-example TRACe scores."""
        results = []
        for ex in examples:
            if use_ground_truth and ex.get("relevance") is not None:
                results.append(self.scores_from_ground_truth(ex))
            else:
                results.append(
                    self.estimate_scores(
                        question=ex["question"],
                        response=ex["response"],
                        documents=ex["documents"],
                    )
                )
        return results

    # ---- aggregate statistics ----

    @staticmethod
    def aggregate(scores: list[TRACeScores]) -> dict:
        """Compute mean / std / counts over a list of TRACe score objects."""
        rels = [s.relevance for s in scores if s.relevance is not None]
        utils = [s.utilization for s in scores if s.utilization is not None]
        comps = [s.completeness for s in scores if s.completeness is not None]
        adhs = [1 if s.adherence else 0 for s in scores if s.adherence is not None]

        def _stats(vals):
            if not vals:
                return {"mean": None, "std": None, "count": 0}
            arr = np.array(vals, dtype=float)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "count": len(vals)}

        return {
            "relevance": _stats(rels),
            "utilization": _stats(utils),
            "completeness": _stats(comps),
            "adherence": _stats(adhs),
            "hallucination_rate": {
                "mean": 1.0 - np.mean(adhs) if adhs else None,
                "count": len(adhs),
            },
        }

    # ---- alignment with ground truth (for benchmarking evaluators) ----

    @staticmethod
    def compute_alignment(
        predicted: list[TRACeScores], ground_truth: list[TRACeScores]
    ) -> dict:
        """
        Compute alignment between predicted and ground-truth TRACe scores.
        Returns RMSE for continuous metrics and AUROC for adherence.
        """
        results: dict = {}

        for metric in ("relevance", "utilization", "completeness"):
            preds = []
            golds = []
            for p, g in zip(predicted, ground_truth):
                pv = getattr(p, metric)
                gv = getattr(g, metric)
                if pv is not None and gv is not None:
                    preds.append(pv)
                    golds.append(gv)
            if preds:
                rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(golds)) ** 2)))
                results[f"{metric}_rmse"] = rmse
            else:
                results[f"{metric}_rmse"] = None

        # Adherence — AUROC
        pred_adh = []
        gold_adh = []
        for p, g in zip(predicted, ground_truth):
            if p.adherence is not None and g.adherence is not None:
                pred_adh.append(int(p.adherence))
                gold_adh.append(int(g.adherence))
        if pred_adh and len(set(gold_adh)) > 1:
            results["adherence_auroc"] = float(roc_auc_score(gold_adh, pred_adh))
        else:
            results["adherence_auroc"] = None

        return results
