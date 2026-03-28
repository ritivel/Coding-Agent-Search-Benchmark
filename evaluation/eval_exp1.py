#!/usr/bin/env python3
"""
Evaluate Experiment 1 RAG pipeline on RAGBench.

Pipeline: Recursive character chunking -> ChromaDB + BM25/RRF hybrid -> Cross-encoder reranking
This replicates Experiment 1's advanced retrieval approach and evaluates it
against RAGBench QA.
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
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

from evaluation.data_loader import load_ragbench
from evaluation.evaluator import RAGEvaluator
from evaluation.report import ComparisonReport

logging.basicConfig(level=logging.WARNING)

SUBSETS = ["hotpotqa", "covidqa"]
SAMPLE_SIZE = 25
MODEL_NAME = "experiment_1_hybrid_rrf_rerank"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
RERANK_CANDIDATES = 20
RRF_K = 60
HYBRID_ALPHA = 0.7


def recursive_chunk(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def _merge(parts, sep):
        merged = []
        current_parts = []
        current_len = 0
        for part in parts:
            part_len = len(part)
            sep_len = len(sep) if current_parts else 0
            if current_len + part_len + sep_len > chunk_size and current_parts:
                merged.append(sep.join(current_parts))
                overlap_parts = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) + len(sep) > chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + len(sep)
                current_parts = overlap_parts
                current_len = sum(len(p) for p in current_parts) + len(sep) * max(0, len(current_parts) - 1)
            current_parts.append(part)
            current_len += part_len + sep_len
        if current_parts:
            merged.append(sep.join(current_parts))
        return merged

    def _split(text, seps):
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        sep = seps[0] if seps else " "
        remaining = seps[1:] if len(seps) > 1 else []
        parts = text.split(sep) if sep else [text]
        final = []
        good = []
        for s in parts:
            if len(s) <= chunk_size:
                good.append(s)
            else:
                if good:
                    final.extend(_merge(good, sep))
                    good = []
                if remaining:
                    final.extend(_split(s, remaining))
                else:
                    final.append(s[:chunk_size])
        if good:
            final.extend(_merge(good, sep))
        return [c for c in final if c.strip()]

    return _split(text, separators)


class Experiment1Adapter:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve_and_respond(self, question: str, documents: list[str], top_k: int = TOP_K) -> str:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(recursive_chunk(doc, CHUNK_SIZE, CHUNK_OVERLAP))

        if not all_chunks:
            return ""

        chunk_embs = self.embed_model.encode(
            all_chunks, show_progress_bar=False, convert_to_numpy=True
        )

        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("temp")
        except Exception:
            pass
        collection = client.create_collection(
            name="temp", metadata={"hnsw:space": "cosine"}
        )

        ids = [f"c_{i}" for i in range(len(all_chunks))]
        collection.add(
            ids=ids, embeddings=chunk_embs.tolist(), documents=all_chunks
        )

        q_emb = self.embed_model.encode(
            [question], show_progress_bar=False, convert_to_numpy=True
        )
        n_results = min(RERANK_CANDIDATES, len(all_chunks))
        vector_results = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n_results,
            include=["documents", "distances"],
        )

        vector_docs = vector_results["documents"][0] if vector_results["documents"] else []

        tokenized = [doc.lower().split() for doc in all_chunks]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(question.lower().split())
        bm25_ranked = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:RERANK_CANDIDATES]

        rrf_scores = {}
        for rank, doc_text in enumerate(vector_docs):
            rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + HYBRID_ALPHA / (RRF_K + rank + 1)

        for rank, idx in enumerate(bm25_ranked):
            doc_text = all_chunks[idx]
            rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1 - HYBRID_ALPHA) / (RRF_K + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused_texts = [text for text, _ in fused[:RERANK_CANDIDATES]]

        if not fused_texts:
            return ""

        if len(fused_texts) > 1:
            pairs = [(question, t) for t in fused_texts]
            rerank_scores = self.reranker.predict(pairs)
            reranked = sorted(
                zip(fused_texts, rerank_scores), key=lambda x: x[1], reverse=True
            )
            retrieved = [text for text, _ in reranked[:top_k]]
        else:
            retrieved = fused_texts[:top_k]

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

    adapter = Experiment1Adapter()

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
