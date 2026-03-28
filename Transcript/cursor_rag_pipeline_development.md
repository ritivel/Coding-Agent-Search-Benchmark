# RAG pipeline development
_Exported on 28/03/2026 at 21:34:39 GMT+5:30 from Cursor (2.6.20)_

---

**User**

This is experiment 2.

Build a RAG pipeline that takes handwritten document images as input, extracts text, and produces a structured digital document as output. Use parallel to find the best methodology to create a RAG. Create a the experiment_2  folder for this experiment.And also create a README file explaining all the steps that you had taken.To develop this particular RAG pipeline

curl -X POST https://api.parallel.ai/v1beta/search \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: Rqs1pcuTrq50a7z8fjelr_PeTBqPAjd8H-7CfeK3' \
  -d '{
  "mode": "one-shot",
  "search_queries": null,
  "max_results": 10,
  "objective": "Find the latest information regarding various web tools or approaches to develop or write code related to the project that I am writing."
}'


Use the parallel ai curl to find the best methodology to build the RAG system, or find the best tool to build the RAG system, and then create that RAG system or implement that RAG system.

---

**Cursor**

I'll start by understanding the existing experiments for context, then use Parallel AI to research the best approach for building a handwritten document RAG pipeline.

Now let me use Parallel AI to research the best methodology and tools for building a handwritten document RAG pipeline, and create the experiment_2 folder simultaneously.

Let me read the Parallel AI research results.

Excellent. I now have comprehensive research from Parallel AI. Based on the findings, Experiment 2 will significantly improve over Experiment 0 by using **Surya OCR** (modern, 90+ languages, layout-aware), **ChromaDB** (better metadata handling), **semantic chunking**, **cross-encoder re-ranking**, and **hybrid search**. Let me build this.

Now let me create all the pipeline files. I'll dispatch parallel agents for the independent file creation tasks.

Now let me create all the pipeline modules in parallel using subagents for speed.

Good, all pipeline modules are created. Let me fix the missing `import cv2` in the OCR file, create `requirements.txt`, and then create the comprehensive README.

Now let me verify all files exist and create the comprehensive README.

All 13 files are in place. Now let me create the comprehensive README.

Now let me do a final verification — check the image_preprocessor was written correctly and verify all files are syntactically valid.

All 12 Python files pass syntax validation. Let me do one final check on the complete file listing.

Experiment 2 is complete. Here's a summary of what was built:

## Experiment 2 — Handwritten Document RAG Pipeline

**14 files created** (2,038 total lines of code) in `experiment_2/`:

### Research Phase (Parallel AI)
Three Parallel AI queries were used to find the best 2026 methodology:
1. OCR tools comparison (Surya, TrOCR, PaddleOCR, olmOCR, Qwen-VL)
2. RAG implementation best practices (chunking, vector stores, hybrid search)
3. Multimodal RAG advances (ColPali, M3DocRAG, vision-language models)

### Pipeline Architecture (7 Stages)

| Stage | Module | Key Technology |
|-------|--------|---------------|
| 1 | `image_preprocessor.py` | OpenCV (denoise, CLAHE, binarize, deskew, line segmentation) |
| 2 | `handwriting_ocr.py` | **Surya OCR** (90+ languages, layout-aware) with TrOCR fallback |
| 3 | `text_postprocessor.py` | Unicode normalization, OCR error correction, artifact removal |
| 4a | `embeddings.py` | **Semantic chunking** (topic-boundary splitting) + MiniLM embeddings |
| 4b | `vector_store.py` | **ChromaDB** (persistent, metadata-aware, hybrid search) |
| 5 | `reranker.py` | **Cross-encoder re-ranking** (ms-marco-MiniLM) |
| 6 | `document_structurer.py` | Heading/list detection, output as JSON/Markdown/plain text |

### Key Improvements over Experiment 0
- **Surya OCR** (90+ languages) replaces TrOCR (English-only)
- **Semantic chunking** replaces fixed-size chunking (15-30% better retrieval)
- **ChromaDB** replaces FAISS (persistent, metadata-aware)
- **Hybrid search** (vector 70% + keyword 30%) for better recall
- **Cross-encoder re-ranking** for 10-20% precision improvement

The comprehensive `README.md` documents every design decision, research source, and the rationale behind each technology choice.

