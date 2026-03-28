#!/usr/bin/env python3
"""Combine evaluation results from all four experiments into a unified report."""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULT_FILES = [
    "results_experiment_0_faiss_wordwindow.json",
    "results_experiment_1_hybrid_rrf_rerank.json",
    "results_experiment_2_semantic_hybrid_rerank.json",
    "results_experiment_3_langchain_faiss.json",
]

FRIENDLY_NAMES = {
    "experiment_0_faiss_wordwindow": "Exp0: FAISS + Word-Window",
    "experiment_1_hybrid_rrf_rerank": "Exp1: ChromaDB + BM25/RRF + Rerank",
    "experiment_2_semantic_hybrid_rerank": "Exp2: Semantic + Hybrid + Rerank",
    "experiment_3_langchain_faiss": "Exp3: LangChain + FAISS",
}


def _fmt(val, decimals=4):
    if val is None:
        return "    --"
    return f"{val:.{decimals}f}"


def main():
    eval_dir = Path(__file__).resolve().parent

    print(f"\n{'='*80}")
    print("  COMBINED RAG PIPELINE EVALUATION REPORT")
    print(f"  Benchmark: RAGBench (TRACe + Traditional NLG Metrics)")
    print(f"{'='*80}\n")

    all_results = {}
    for fname in RESULT_FILES:
        fpath = eval_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            all_results.update(data)
            print(f"  [OK] {fname}")
        else:
            print(f"  [MISSING] {fname}")

    if not all_results:
        print("\nNo results found. Run the individual evaluation scripts first.")
        return

    print(f"\n{'='*80}")
    print("  TRACe METRICS (Heuristic Estimation)")
    print(f"{'='*80}\n")

    header = f"{'Pipeline':<42} {'Relevance':>10} {'Utilization':>12} {'Complete':>10} {'Adherence':>10} {'Halluc%':>8}"
    print(header)
    print("-" * len(header))

    for model_name, data in sorted(all_results.items()):
        trace = data.get("trace_aggregate", {})
        display_name = FRIENDLY_NAMES.get(model_name, model_name)
        rel = trace.get("relevance", {}).get("mean")
        util = trace.get("utilization", {}).get("mean")
        comp = trace.get("completeness", {}).get("mean")
        adh = trace.get("adherence", {}).get("mean")
        hall = trace.get("hallucination_rate", {}).get("mean")
        print(
            f"{display_name:<42} {_fmt(rel):>10} {_fmt(util):>12} {_fmt(comp):>10} {_fmt(adh):>10} {_fmt(hall, 2):>8}"
        )

    print(f"\n{'='*80}")
    print("  TRADITIONAL NLG METRICS")
    print(f"{'='*80}\n")

    header = f"{'Pipeline':<42} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BLEU':>8} {'F1':>8} {'EM':>8}"
    print(header)
    print("-" * len(header))

    for model_name, data in sorted(all_results.items()):
        trad = data.get("traditional_aggregate", {})
        display_name = FRIENDLY_NAMES.get(model_name, model_name)
        r1 = trad.get("rouge1", {}).get("mean")
        r2 = trad.get("rouge2", {}).get("mean")
        rl = trad.get("rougeL", {}).get("mean")
        bleu = trad.get("bleu", {}).get("mean")
        f1 = trad.get("f1", {}).get("mean")
        em = trad.get("exact_match", {}).get("mean")
        print(
            f"{display_name:<42} {_fmt(r1):>8} {_fmt(r2):>8} {_fmt(rl):>8} {_fmt(bleu):>8} {_fmt(f1):>8} {_fmt(em):>8}"
        )

    print(f"\n{'='*80}")
    print("  OVERALL RANKING (Composite TRACe Score)")
    print(f"{'='*80}\n")

    rankings = []
    for model_name, data in all_results.items():
        trace = data.get("trace_aggregate", {})
        vals = []
        for m in ("relevance", "utilization", "completeness", "adherence"):
            v = trace.get(m, {}).get("mean")
            if v is not None:
                vals.append(v)
        composite = sum(vals) / len(vals) if vals else 0
        display_name = FRIENDLY_NAMES.get(model_name, model_name)
        rankings.append((display_name, composite))

    rankings.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(rankings, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank}: {name} — Composite: {score:.4f}{marker}")

    print(f"\n{'='*80}")
    print("  PER-SUBSET BREAKDOWN")
    print(f"{'='*80}\n")

    all_subsets = set()
    for data in all_results.values():
        all_subsets.update(data.get("per_subset_trace", {}).keys())

    for subset in sorted(all_subsets):
        print(f"  [{subset}]")
        for model_name, data in sorted(all_results.items()):
            display_name = FRIENDLY_NAMES.get(model_name, model_name)
            agg = data.get("per_subset_trace", {}).get(subset, {})
            rel = agg.get("relevance", {}).get("mean")
            adh = agg.get("adherence", {}).get("mean")
            util = agg.get("utilization", {}).get("mean")
            print(
                f"    {display_name:<40} R={_fmt(rel,3)} U={_fmt(util,3)} A={_fmt(adh,3)}"
            )
        print()

    combined_path = eval_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
