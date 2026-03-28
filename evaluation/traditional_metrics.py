"""
Traditional NLG / QA evaluation metrics:
  ROUGE-1, ROUGE-2, ROUGE-L, BLEU, token-level F1, Exact Match, BERTScore.

These complement TRACe by measuring surface-level similarity between the
generated response and a reference (ground-truth) answer, when available.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TraditionalScores:
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bleu: Optional[float] = None
    f1: Optional[float] = None
    exact_match: Optional[float] = None
    bert_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Normalisation helpers (SQuAD-style)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> list[str]:
    return _normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Individual metric implementations
# ---------------------------------------------------------------------------

def compute_exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(reference))


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F-scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        return {"rouge1": None, "rouge2": None, "rougeL": None}


def compute_bleu(prediction: str, reference: str) -> Optional[float]:
    """Sentence-level BLEU (smoothed)."""
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = _get_tokens(reference)
        pred_tokens = _get_tokens(prediction)
        if not ref_tokens or not pred_tokens:
            return 0.0
        smooth = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))
    except ImportError:
        return None


def compute_bert_score(
    predictions: list[str], references: list[str]
) -> list[float]:
    """BERTScore F1 for a batch. Returns per-example F1 list."""
    try:
        from bert_score import score as bscore
        _, _, f1 = bscore(predictions, references, lang="en", verbose=False)
        return f1.tolist()
    except ImportError:
        return [None] * len(predictions)


# ---------------------------------------------------------------------------
# Aggregated class
# ---------------------------------------------------------------------------

class TraditionalMetrics:
    """Compute traditional NLG metrics for (prediction, reference) pairs."""

    def __init__(self, compute_bertscore: bool = False):
        self.compute_bertscore = compute_bertscore

    def score_single(
        self, prediction: str, reference: str, bert_score_val: Optional[float] = None
    ) -> TraditionalScores:
        if not reference:
            return TraditionalScores()

        rouge = compute_rouge(prediction, reference)
        return TraditionalScores(
            rouge1=rouge["rouge1"],
            rouge2=rouge["rouge2"],
            rougeL=rouge["rougeL"],
            bleu=compute_bleu(prediction, reference),
            f1=compute_f1(prediction, reference),
            exact_match=compute_exact_match(prediction, reference),
            bert_score=bert_score_val,
        )

    def score_batch(
        self, predictions: list[str], references: list[str]
    ) -> list[TraditionalScores]:
        bert_scores = (
            compute_bert_score(predictions, references)
            if self.compute_bertscore
            else [None] * len(predictions)
        )

        results = []
        for pred, ref, bs in zip(predictions, references, bert_scores):
            results.append(self.score_single(pred, ref, bert_score_val=bs))
        return results

    @staticmethod
    def aggregate(scores: list[TraditionalScores]) -> dict:
        out: dict = {}
        for metric in ("rouge1", "rouge2", "rougeL", "bleu", "f1", "exact_match", "bert_score"):
            vals = [getattr(s, metric) for s in scores if getattr(s, metric) is not None]
            if vals:
                arr = np.array(vals, dtype=float)
                out[metric] = {"mean": float(arr.mean()), "std": float(arr.std()), "count": len(vals)}
            else:
                out[metric] = {"mean": None, "std": None, "count": 0}
        return out
