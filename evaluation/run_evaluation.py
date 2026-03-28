#!/usr/bin/env python3
"""
CLI entry-point for the RAG evaluation framework.

Usage examples
--------------
# Evaluate RAGBench ground-truth labels on 2 subsets (quick smoke test)
python -m evaluation.run_evaluation --subsets hotpotqa covidqa --split test --sample 50

# Full evaluation on all 12 subsets
python -m evaluation.run_evaluation --split test

# Export results
python -m evaluation.run_evaluation --subsets hotpotqa --output-json results.json --output-csv results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation.config import RAGBENCH_SUBSETS, DEFAULT_SPLIT
from evaluation.data_loader import load_ragbench
from evaluation.evaluator import RAGEvaluator
from evaluation.report import ComparisonReport


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG Evaluation Framework — RAGBench + TRACe metrics"
    )
    p.add_argument(
        "--subsets", nargs="+", default=None,
        help="RAGBench subsets to evaluate (default: all 12).",
    )
    p.add_argument(
        "--split", default=DEFAULT_SPLIT,
        choices=["train", "validation", "test"],
    )
    p.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N examples per subset (for quick testing).",
    )
    p.add_argument(
        "--model-name", default="ragbench_ground_truth",
        help="Label for the model being evaluated.",
    )
    p.add_argument(
        "--bertscore", action="store_true",
        help="Compute BERTScore (slower, needs GPU).",
    )
    p.add_argument(
        "--output-json", default=None,
        help="Path to write JSON results.",
    )
    p.add_argument(
        "--output-csv", default=None,
        help="Path to write CSV results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print("  RAG Evaluation Framework")
    print(f"  Subsets : {args.subsets or 'all 12'}")
    print(f"  Split   : {args.split}")
    print(f"  Sample  : {args.sample or 'full'}")
    print(f"{'='*60}\n")

    # 1. Load data
    examples = load_ragbench(
        subsets=args.subsets,
        split=args.split,
        sample_size=args.sample,
    )
    print(f"Loaded {len(examples)} examples.\n")

    # 2. Evaluate
    evaluator = RAGEvaluator(compute_bertscore=args.bertscore)
    evaluation = evaluator.evaluate(
        examples,
        model_name=args.model_name,
        use_ground_truth_trace=True,
    )

    # 3. Report
    report = ComparisonReport()
    report.add(evaluation)

    print(report.full_report())

    if args.output_json:
        report.to_json(args.output_json)
        print(f"\nJSON results written to {args.output_json}")

    if args.output_csv:
        report.to_csv(args.output_csv)
        print(f"CSV results written to {args.output_csv}")


if __name__ == "__main__":
    main()
