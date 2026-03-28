"""
Load RAGBench data from HuggingFace and normalise it into a uniform schema
that the evaluation framework can consume regardless of subset.

Each example is normalised to:
    {
        "subset":     str,       # e.g. "hotpotqa"
        "domain":     str,       # e.g. "general_knowledge"
        "question":   str,
        "documents":  list[str], # retrieved context docs
        "response":   str,       # LLM-generated response
        "ground_truth_response": str | None,
        # --- TRACe ground-truth labels ---
        "relevance":     float | None,
        "utilization":   float | None,
        "completeness":  float | None,
        "adherence":     bool  | None,
        # raw annotation blob for fine-grained analysis
        "annotations":   dict  | None,
    }
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from .config import RAGBENCH_HF_REPO, RAGBENCH_SUBSETS, DOMAIN_MAP, DEFAULT_SPLIT


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalise_example(raw: dict, subset: str) -> dict:
    """Map a raw HuggingFace row to the canonical schema."""
    documents = raw.get("documents") or raw.get("contexts") or raw.get("context") or []
    if isinstance(documents, str):
        documents = [documents]

    response = raw.get("response") or raw.get("answer") or raw.get("generated_answer") or ""

    ground_truth = raw.get("gold_answer") or raw.get("reference_answer") or raw.get("ground_truth") or None

    relevance = _safe_float(raw.get("relevance"))
    utilization = _safe_float(raw.get("utilization"))
    completeness = _safe_float(raw.get("completeness"))
    adherence_raw = raw.get("adherence") if raw.get("adherence") is not None else raw.get("overall_supported")
    adherence = None
    if adherence_raw is not None:
        adherence = bool(adherence_raw)

    annotations = raw.get("annotations") or raw.get("annotation") or None

    return {
        "subset": subset,
        "domain": DOMAIN_MAP.get(subset, "unknown"),
        "question": raw.get("question") or raw.get("query") or "",
        "documents": documents,
        "response": response,
        "ground_truth_response": ground_truth,
        "relevance": relevance,
        "utilization": utilization,
        "completeness": completeness,
        "adherence": adherence,
        "annotations": annotations,
    }


def load_ragbench(
    subsets: list[str] | None = None,
    split: str = DEFAULT_SPLIT,
    sample_size: int | None = None,
) -> list[dict]:
    """
    Load and normalise RAGBench examples.

    Parameters
    ----------
    subsets : list of subset names, or None for all 12.
    split   : "train", "validation", or "test".
    sample_size : if set, randomly sample this many examples per subset.

    Returns
    -------
    List of normalised example dicts.
    """
    subsets = subsets or RAGBENCH_SUBSETS
    all_examples: list[dict] = []

    for subset in tqdm(subsets, desc="Loading RAGBench subsets"):
        ds = load_dataset(RAGBENCH_HF_REPO, subset, split=split)

        if sample_size is not None and len(ds) > sample_size:
            ds = ds.shuffle(seed=42).select(range(sample_size))

        for row in ds:
            all_examples.append(_normalise_example(row, subset))

    return all_examples


def load_single_subset(
    subset: str,
    split: str = DEFAULT_SPLIT,
    sample_size: int | None = None,
) -> list[dict]:
    """Convenience wrapper for a single subset."""
    return load_ragbench(subsets=[subset], split=split, sample_size=sample_size)


def get_examples_by_domain(examples: list[dict]) -> dict[str, list[dict]]:
    """Group examples by domain."""
    by_domain: dict[str, list[dict]] = {}
    for ex in examples:
        by_domain.setdefault(ex["domain"], []).append(ex)
    return by_domain


def get_examples_by_subset(examples: list[dict]) -> dict[str, list[dict]]:
    """Group examples by subset name."""
    by_subset: dict[str, list[dict]] = {}
    for ex in examples:
        by_subset.setdefault(ex["subset"], []).append(ex)
    return by_subset
