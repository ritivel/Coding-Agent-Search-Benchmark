#!/usr/bin/env python3
"""
Evaluate Experiment 2 RAG pipeline on RAGBench.

Pipeline: Semantic chunking -> ChromaDB hybrid (70% vector + 30% keyword) -> Cross-encoder reranking
This replicates Experiment 2's semantic-similarity-based chunking and hybrid
retrieval approach and evaluates it against RAGBench QA.
"""

import sys
import os
import json
import re
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from evaluation.data_loader import load_ragbench
from evaluation.evaluator import RAGEvaluator
from evaluation.report import ComparisonReport

logging.basicConfig(level=logging.WARNING)

SUBSETS = ["hotpotqa", "covidqa"]
SAMPLE_SIZE = 25
MODEL_NAME = "experiment_2_semantic_hybrid_rerank"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHUNK_SIZE = 512
OVERLAP = 64
SEMANTIC_THRESHOLD = 0.75
TOP_K = 5
VECTOR_WEIGHT = 0.7


def semantic_chunk(
    text: str,
    model: SentenceTransformer,
    max_chunk_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.75,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    chunks = []
    current_chunk = [sentences[0]]
    current_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i], embeddings[i - 1]))
        sent_words = len(sentences[i].split())

        if sim >= threshold and current_words + sent_words <= max_chunk_size:
            current_chunk.append(sentences[i])
            current_words += sent_words
        else:
            chunks.append(" ".join(current_chunk))
            if overlap > 0 and current_chunk:
                tail = current_chunk[-1]
                if len(tail.split()) <= overlap:
                    current_chunk = [tail, sentences[i]]
                else:
                    current_chunk = [sentences[i]]
                current_words = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk = [sentences[i]]
                current_words = sent_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


class Experiment2Adapter:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve_and_respond(self, question: str, documents: list[str], top_k: int = TOP_K) -> str:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(
                semantic_chunk(doc, self.embed_model, MAX_CHUNK_SIZE, OVERLAP, SEMANTIC_THRESHOLD)
            )

        if not all_chunks:
            return ""

        chunk_embs = self.embed_model.encode(
            all_chunks, show_progress_bar=False, normalize_embeddings=True
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

        n_results = min(top_k * 2, len(all_chunks))

        q_emb = self.embed_model.encode(
            [question], show_progress_bar=False, normalize_embeddings=True
        )
        vector_results = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n_results,
            include=["documents", "distances"],
        )
        vector_docs = vector_results["documents"][0] if vector_results["documents"] else []
        vector_dists = vector_results["distances"][0] if vector_results["distances"] else []

        keyword_results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "distances"],
        )
        keyword_docs = keyword_results["documents"][0] if keyword_results["documents"] else []
        keyword_dists = keyword_results["distances"][0] if keyword_results["distances"] else []

        keyword_weight = 1.0 - VECTOR_WEIGHT
        combined = {}
        for doc, dist in zip(vector_docs, vector_dists):
            combined[doc] = combined.get(doc, 0) + (1.0 - dist) * VECTOR_WEIGHT

        for doc, dist in zip(keyword_docs, keyword_dists):
            combined[doc] = combined.get(doc, 0) + (1.0 - dist) * keyword_weight

        fused = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        fused_texts = [text for text, _ in fused[: top_k * 3]]

        if not fused_texts:
            return ""

        if len(fused_texts) > 1:
            pairs = [(question, t) for t in fused_texts]
            rerank_scores = self.reranker.predict(pairs)
            min_s, max_s = min(rerank_scores), max(rerank_scores)
            range_s = max_s - min_s if max_s != min_s else 1.0
            norm_scores = [(s - min_s) / range_s for s in rerank_scores]

            reranked = sorted(
                zip(fused_texts, norm_scores), key=lambda x: x[1], reverse=True
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

    adapter = Experiment2Adapter()

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
