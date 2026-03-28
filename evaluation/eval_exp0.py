#!/usr/bin/env python3
"""
Evaluate Experiment 0 RAG pipeline on RAGBench.

Pipeline: Word-window chunking -> all-MiniLM-L6-v2 embeddings -> FAISS IndexFlatIP
This replicates Experiment 0's retrieval approach (simple FAISS vector search
with word-level sliding window chunking) and evaluates it against RAGBench QA.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from evaluation.data_loader import load_ragbench
from evaluation.evaluator import RAGEvaluator
from evaluation.report import ComparisonReport

logging.basicConfig(level=logging.WARNING)

SUBSETS = ["hotpotqa", "covidqa"]
SAMPLE_SIZE = 25
MODEL_NAME = "experiment_0_faiss_wordwindow"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5


def chunk_word_window(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        step = max(chunk_size - overlap, 1)
        start += step
    return chunks


class Experiment0Adapter:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    def retrieve_and_respond(self, question: str, documents: list[str], top_k: int = TOP_K) -> str:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_word_window(doc, CHUNK_SIZE, CHUNK_OVERLAP))

        if not all_chunks:
            return ""

        chunk_embs = self.model.encode(
            all_chunks, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(chunk_embs)

        index = faiss.IndexFlatIP(self.dim)
        index.add(chunk_embs)

        q_emb = self.model.encode(
            [question], show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(q_emb)

        k = min(top_k, index.ntotal)
        scores, indices = index.search(q_emb, k)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                retrieved.append(all_chunks[idx])

        return " ".join(retrieved)


def main():
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Evaluating: {MODEL_NAME}")
    print(f"  Subsets: {SUBSETS}")
    print(f"  Sample: {SAMPLE_SIZE} per subset")
    print(f"{'='*60}\n")

    examples = load_ragbench(subsets=SUBSETS, split="test", sample_size=SAMPLE_SIZE)
    print(f"Loaded {len(examples)} examples.\n")

    for ex in examples:
        if not ex.get("ground_truth_response"):
            ex["ground_truth_response"] = ex.get("response", "")

    adapter = Experiment0Adapter()

    responses = []
    for ex in tqdm(examples, desc=f"Running {MODEL_NAME}"):
        resp = adapter.retrieve_and_respond(
            question=ex["question"],
            documents=ex["documents"],
        )
        responses.append(resp)

    evaluator = RAGEvaluator()
    evaluation = evaluator.evaluate(
        examples,
        model_name=MODEL_NAME,
        model_responses=responses,
        use_ground_truth_trace=False,
    )

    report = ComparisonReport()
    report.add(evaluation)

    report_text = report.full_report()
    print(report_text)

    output_file = PROJECT_ROOT / "evaluation" / f"results_{MODEL_NAME}.json"
    report.to_json(str(output_file))

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
