"""
Comparison report generator — side-by-side analysis of multiple RAG models.

Produces:
  - Console-friendly tables (via tabulate)
  - CSV export
  - JSON export
  - Per-subset breakdowns
"""

from __future__ import annotations

import csv
import json
import os
from io import StringIO
from typing import Optional

import numpy as np

from .evaluator import ModelEvaluation


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


class ComparisonReport:
    """Build and render comparison reports across multiple model evaluations."""

    def __init__(self):
        self.evaluations: list[ModelEvaluation] = []

    def add(self, evaluation: ModelEvaluation) -> None:
        self.evaluations.append(evaluation)

    # ------------------------------------------------------------------
    # TRACe summary table
    # ------------------------------------------------------------------

    def trace_summary_table(self) -> str:
        """
        Render a table comparing aggregate TRACe metrics across models.
        Columns: Model | Relevance | Utilization | Completeness | Adherence | Hallucination%
        """
        try:
            from tabulate import tabulate
        except ImportError:
            return self._trace_summary_plain()

        headers = [
            "Model",
            "Relevance (μ±σ)",
            "Utilization (μ±σ)",
            "Completeness (μ±σ)",
            "Adherence (μ±σ)",
            "Hallucination %",
        ]
        rows = []
        for ev in self.evaluations:
            t = ev.trace_aggregate
            rows.append([
                ev.model_name,
                self._mu_sigma(t.get("relevance", {})),
                self._mu_sigma(t.get("utilization", {})),
                self._mu_sigma(t.get("completeness", {})),
                self._mu_sigma(t.get("adherence", {})),
                _fmt(t.get("hallucination_rate", {}).get("mean"), 2) if t.get("hallucination_rate", {}).get("mean") is not None else "—",
            ])

        return tabulate(rows, headers=headers, tablefmt="grid")

    def _trace_summary_plain(self) -> str:
        lines = []
        for ev in self.evaluations:
            t = ev.trace_aggregate
            lines.append(f"\n=== {ev.model_name} ===")
            for metric in ("relevance", "utilization", "completeness", "adherence"):
                lines.append(f"  {metric}: {self._mu_sigma(t.get(metric, {}))}")
            hr = t.get("hallucination_rate", {}).get("mean")
            lines.append(f"  hallucination_rate: {_fmt(hr, 2)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Traditional metrics summary
    # ------------------------------------------------------------------

    def traditional_summary_table(self) -> str:
        try:
            from tabulate import tabulate
        except ImportError:
            return "(tabulate not installed)"

        headers = ["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "F1", "EM", "BERTScore"]
        rows = []
        for ev in self.evaluations:
            t = ev.traditional_aggregate
            rows.append([
                ev.model_name,
                *[_fmt(t.get(m, {}).get("mean")) for m in
                  ("rouge1", "rouge2", "rougeL", "bleu", "f1", "exact_match", "bert_score")],
            ])
        return tabulate(rows, headers=headers, tablefmt="grid")

    # ------------------------------------------------------------------
    # Per-subset breakdown
    # ------------------------------------------------------------------

    def per_subset_table(self) -> str:
        try:
            from tabulate import tabulate
        except ImportError:
            return "(tabulate not installed)"

        all_subsets = sorted({
            s for ev in self.evaluations for s in ev.per_subset_trace
        })

        headers = ["Subset"] + [ev.model_name for ev in self.evaluations]
        rows = []
        for subset in all_subsets:
            row = [subset]
            for ev in self.evaluations:
                agg = ev.per_subset_trace.get(subset, {})
                rel = agg.get("relevance", {}).get("mean")
                adh = agg.get("adherence", {}).get("mean")
                row.append(f"R={_fmt(rel,2)} A={_fmt(adh,2)}")
            rows.append(row)

        return tabulate(rows, headers=headers, tablefmt="grid")

    # ------------------------------------------------------------------
    # Ranking table
    # ------------------------------------------------------------------

    def ranking_table(self) -> str:
        """Rank models by a composite score: mean of all TRACe means."""
        try:
            from tabulate import tabulate
        except ImportError:
            return "(tabulate not installed)"

        scores = []
        for ev in self.evaluations:
            t = ev.trace_aggregate
            vals = []
            for m in ("relevance", "utilization", "completeness", "adherence"):
                v = t.get(m, {}).get("mean")
                if v is not None:
                    vals.append(v)
            composite = float(np.mean(vals)) if vals else 0.0
            scores.append((ev.model_name, composite))

        scores.sort(key=lambda x: x[1], reverse=True)
        headers = ["Rank", "Model", "Composite TRACe"]
        rows = [[i + 1, name, _fmt(sc, 4)] for i, (name, sc) in enumerate(scores)]
        return tabulate(rows, headers=headers, tablefmt="grid")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self, path: Optional[str] = None) -> str:
        payload = {}
        for ev in self.evaluations:
            payload[ev.model_name] = {
                "trace_aggregate": ev.trace_aggregate,
                "traditional_aggregate": ev.traditional_aggregate,
                "per_subset_trace": ev.per_subset_trace,
                "per_domain_trace": ev.per_domain_trace,
            }
        blob = json.dumps(payload, indent=2, default=str)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(blob)
        return blob

    def to_csv(self, path: str) -> None:
        """Export per-model aggregate TRACe metrics to CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "relevance_mean", "utilization_mean",
                "completeness_mean", "adherence_mean", "hallucination_rate",
            ])
            for ev in self.evaluations:
                t = ev.trace_aggregate
                writer.writerow([
                    ev.model_name,
                    t.get("relevance", {}).get("mean"),
                    t.get("utilization", {}).get("mean"),
                    t.get("completeness", {}).get("mean"),
                    t.get("adherence", {}).get("mean"),
                    t.get("hallucination_rate", {}).get("mean"),
                ])

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> str:
        sections = [
            "=" * 70,
            "  RAG EVALUATION COMPARISON REPORT  (TRACe + Traditional Metrics)",
            "=" * 70,
            "",
            "--- TRACe Metrics Summary ---",
            self.trace_summary_table(),
            "",
            "--- Traditional Metrics Summary ---",
            self.traditional_summary_table(),
            "",
            "--- Per-Subset Breakdown (Relevance & Adherence) ---",
            self.per_subset_table(),
            "",
            "--- Model Ranking (Composite TRACe) ---",
            self.ranking_table(),
            "",
            "=" * 70,
        ]
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mu_sigma(stat: dict) -> str:
        m = stat.get("mean")
        s = stat.get("std")
        if m is None:
            return "—"
        if s is not None:
            return f"{m:.4f} ± {s:.4f}"
        return _fmt(m)
