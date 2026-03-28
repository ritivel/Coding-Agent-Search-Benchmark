# RAGBench Evaluation — Changes & Rationale

## Overview

This document records all changes made to enable RAGBench evaluation of the four RAG pipeline experiments (0–3). The original experiments were designed for **handwritten document OCR + digitisation**, while the evaluation benchmark (RAGBench) is a **text-based question-answering** benchmark. Bridging this gap required creating adapter scripts and addressing several architectural mismatches.

---

## Problem Statement

The four experiments implement different RAG retrieval strategies:

| Experiment | Chunking | Vector Store | Retrieval | Generation |
|------------|----------|-------------|-----------|------------|
| 0 | Word-window (512 words, 64 overlap) | FAISS IndexFlatIP | Cosine similarity | None (rule-based structurer) |
| 1 | Recursive character splitting | ChromaDB | BM25 + Vector RRF fusion + Cross-encoder rerank | None (rule-based structurer) |
| 2 | Semantic chunking (sentence similarity) | ChromaDB | Hybrid (70% vector + 30% keyword) + Cross-encoder rerank | None (rule-based structurer) |
| 3 | LangChain RecursiveCharacterTextSplitter | FAISS (via LangChain) | L2 distance search | LLM (OpenAI/flan-t5-base) |

**Key gap**: There was **no bridge** between these experiment pipelines and the evaluation framework. The `BaseRAGModel` abstract class existed in `evaluation/base_rag.py` but was never subclassed by any experiment.

---

## Changes Made

### 1. Created Evaluation Adapter Scripts

**Files created:**
- `evaluation/eval_exp0.py` — Experiment 0 adapter
- `evaluation/eval_exp1.py` — Experiment 1 adapter
- `evaluation/eval_exp2.py` — Experiment 2 adapter
- `evaluation/eval_exp3.py` — Experiment 3 adapter
- `evaluation/combine_results.py` — Results combiner

**Why**: Each experiment's pipeline was tightly coupled to OCR image processing. RAGBench provides text documents, not images. These adapter scripts replicate each experiment's **retrieval approach** (chunking + indexing + search) while bypassing the OCR stages, enabling direct evaluation on RAGBench.

**Logic**: Each adapter:
1. Takes RAGBench text documents as input (bypassing OCR preprocessing)
2. Chunks the documents using the experiment's specific chunking strategy
3. Indexes chunks using the experiment's vector store technology
4. Queries using the experiment's retrieval mechanism
5. Concatenates retrieved top-k chunks as the response
6. Passes responses to the evaluation framework for TRACe + traditional NLG metrics

### 2. Bypassed OCR Stage

**Why**: RAGBench provides well-formed text passages, not handwritten document images. Running OCR on text would be nonsensical. The OCR stage (TrOCR, Surya, Tesseract) is the input preprocessing step — the evaluation focuses on the retrieval pipeline quality downstream of OCR.

**Logic**: The adapters work directly with text, testing each pipeline's ability to:
- Chunk text effectively for retrieval
- Index and search efficiently
- Retrieve the most relevant passages for a question

### 3. Used Heuristic TRACe Estimation (not ground-truth labels)

**Why**: RAGBench provides pre-annotated TRACe labels (relevance, utilization, completeness, adherence) but these are properties of the **benchmark's own RAG system**, not ours. Using `use_ground_truth_trace=True` would yield identical TRACe scores for all four pipelines, making comparison meaningless.

**Logic**: Setting `use_ground_truth_trace=False` causes the evaluator to **estimate** TRACe metrics from:
- **Relevance**: Question-document token overlap (same for all pipelines since they share documents)
- **Utilization**: What fraction of document content appears in the response (differs per pipeline)
- **Completeness**: What fraction of relevant content is utilized (differs per pipeline)
- **Adherence**: Whether the response is grounded in documents (differs per pipeline)

This makes utilization, completeness, and adherence pipeline-dependent, enabling meaningful comparison.

### 4. Response Generation by Chunk Concatenation

**Why**: Experiments 0–2 have no LLM generation component — they are retrieval-only pipelines with rule-based document structurers. Experiment 3 has LLM capability but requires an API key (OpenAI) or a local model (flan-t5-base) that produces low-quality short outputs.

**Logic**: For a **fair comparison** of retrieval quality across all four pipelines, all adapters use the same response generation strategy: concatenating the top-k retrieved chunks into a single response string. This means:
- Traditional metrics (ROUGE, BLEU, F1) measure how well the retrieved chunks cover the ground-truth answer
- TRACe metrics measure how well the retrieval interacts with the provided documents
- The only variable is the retrieval pipeline, not the generation model

### 5. Per-Example Fresh Indexing

**Why**: Each RAGBench example has its own set of context documents. The vector store must be rebuilt per example to avoid cross-contamination between examples.

**Logic**: Each adapter creates a fresh in-memory index (FAISS or ChromaDB EphemeralClient) for every example, indexes that example's documents, queries, then discards the index. This is computationally more expensive but ensures clean evaluation.

### 6. Experiment 2 — sys.path Fix (Identified Issue)

**Original bug**: `experiment_2/run_pipeline.py` uses `sys.path.insert(0, str(Path(__file__).parent.parent))` which adds the **repository root** to the path, not `experiment_2/`. This causes `import config` to fail since there's no `config.py` at the repo root. Experiments 0 and 1 correctly use `Path(__file__).resolve().parent`.

**Resolution in adapter**: The adapter script is self-contained and does not import from `experiment_2/`, so this bug is sidestepped. The issue is documented here for future reference.

### 7. ChromaDB Collection Collision Fix (Experiments 1 & 2)

**Original bug**: `chromadb.EphemeralClient()` shares state within a single process. Calling `client.create_collection("temp")` on the second RAGBench example fails because the "temp" collection from the first example still exists.

**Fix**: Added a `client.delete_collection("temp")` call (wrapped in try/except) before each `create_collection`, ensuring a clean vector store for every example.

### 8. Ground-Truth Response Fallback for Traditional Metrics

**Original issue**: RAGBench examples set `ground_truth_response` to None because the dataset fields (`gold_answer`, `reference_answer`, `ground_truth`) don't exist in the raw data. This caused all traditional metrics (ROUGE, BLEU, F1, EM) to show "—" (not computable).

**Fix**: When `ground_truth_response` is None, fall back to using the RAGBench `response` field (the original LLM-generated answer) as the reference. This enables traditional metric computation by comparing our pipeline's retrieved chunks against the RAGBench model's answer.

### 9. Experiment 3 — LLM Dependency Removed for Evaluation

**Original design**: Experiment 3's `rag_pipeline.py` imports `config` (bare module) and requires either an OpenAI API key or downloads `google/flan-t5-base` for generation.

**Resolution**: The adapter uses only the retrieval components (embeddings + FAISS indexing + search), bypassing the LLM entirely. This avoids API key requirements and heavy model downloads while still testing Experiment 3's distinctive features: LangChain-style character-level recursive chunking and FAISS L2 distance search.

---

## Evaluation Configuration

All four evaluation scripts use identical parameters for fairness:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Subsets | hotpotqa, covidqa | General knowledge + biomedical domains |
| Sample size | 25 per subset | 50 total examples — fast enough for iterative testing |
| Split | test | Standard evaluation split |
| Top-k retrieval | 5 | Consistent across all pipelines |
| Embedding model | all-MiniLM-L6-v2 | Same model used by all four experiments |
| BERTScore | Disabled | Too slow for iterative evaluation |

---

## How to Run

```bash
# Individual experiments
python evaluation/eval_exp0.py
python evaluation/eval_exp1.py
python evaluation/eval_exp2.py
python evaluation/eval_exp3.py

# Combine all results
python evaluation/combine_results.py
```

Results are saved as JSON files in `evaluation/` with the prefix `results_`.

---

## Expected Differences Between Pipelines

| Feature | Exp 0 | Exp 1 | Exp 2 | Exp 3 |
|---------|-------|-------|-------|-------|
| Chunking granularity | Coarse (512 words) | Medium (recursive character) | Fine (semantic sentence groups) | Medium (500 chars recursive) |
| Keyword matching | None | BM25 | ChromaDB text search | None |
| Hybrid fusion | No | RRF (alpha=0.7) | Weighted (70/30) | No |
| Reranking | No | Cross-encoder | Cross-encoder | No |
| Expected strength | Simple, fast | Best precision via reranking | Best semantic coherence | Good character-level splitting |
