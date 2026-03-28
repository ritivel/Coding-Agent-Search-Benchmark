#!/usr/bin/env python3
"""
Evaluate Experiment 3 RAG pipeline on RAGBench.

Pipeline: LangChain RecursiveCharacterTextSplitter -> FAISS (IndexFlatL2) -> Retrieval
This replicates Experiment 3's character-level recursive chunking with LangChain
and FAISS L2 distance search, and evaluates it against RAGBench QA.
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
MODEL_NAME = "experiment_3_langchain_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


def langchain_recursive_chunk(
    text: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs if doc.page_content.strip()]
    except ImportError:
        separators = ["\n\n", "\n", ". ", " "]

        def _split(txt, seps):
            if len(txt) <= chunk_size:
                return [txt] if txt.strip() else []
            sep = seps[0] if seps else " "
            rest = seps[1:] if len(seps) > 1 else []
            parts = txt.split(sep) if sep else [txt]
            chunks = []
            current = ""
            for p in parts:
                candidate = (current + sep + p) if current else p
                if len(candidate) > chunk_size and current:
                    chunks.append(current)
                    overlap_start = max(0, len(current) - chunk_overlap)
                    current = current[overlap_start:] + sep + p if overlap_start > 0 else p
                else:
                    current = candidate
            if current.strip():
                chunks.append(current)
            final = []
            for c in chunks:
                if len(c) > chunk_size and rest:
                    final.extend(_split(c, rest))
                elif c.strip():
                    final.append(c)
            return final

        return _split(text, separators)


class Experiment3Adapter:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    def retrieve_and_respond(self, question: str, documents: list[str], top_k: int = TOP_K) -> str:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(langchain_recursive_chunk(doc, CHUNK_SIZE, CHUNK_OVERLAP))

        if not all_chunks:
            return ""

        chunk_embs = self.model.encode(
            all_chunks, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")

        index = faiss.IndexFlatL2(self.dim)
        index.add(chunk_embs)

        q_emb = self.model.encode(
            [question], show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")

        k = min(top_k, index.ntotal)
        distances, indices = index.search(q_emb, k)

        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
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

    adapter = Experiment3Adapter()

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
