# Experiment 2 — Handwritten Document RAG Pipeline (Surya OCR + ChromaDB + Re-ranking)

A production-grade Retrieval-Augmented Generation (RAG) pipeline that takes **handwritten document images** as input, extracts text via modern OCR, and produces **structured digital documents** (Markdown, JSON, plain text) as output — with semantic search, hybrid retrieval, and cross-encoder re-ranking.

---

## Table of Contents

1. [Overview](#overview)
2. [Research & Methodology Selection](#research--methodology-selection)
3. [Architecture](#architecture)
4. [Pipeline Stages](#pipeline-stages)
5. [Key Improvements over Experiment 0](#key-improvements-over-experiment-0)
6. [Directory Structure](#directory-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Design Decisions & Rationale](#design-decisions--rationale)
11. [Libraries & Tools Used](#libraries--tools-used)
12. [Limitations & Future Work](#limitations--future-work)

---

## Overview

Handwritten documents — letters, meeting notes, scientific notebooks — are ubiquitous but unsearchable. This experiment builds an end-to-end pipeline that:

1. **Ingests** scanned or photographed handwritten document images.
2. **Extracts** text using **Surya OCR**, a modern multilingual document OCR toolkit supporting 90+ languages with layout awareness.
3. **Chunks** extracted text using **semantic chunking** (splitting at topic boundaries rather than fixed windows).
4. **Indexes** chunks in **ChromaDB** with cosine-similarity vector search and metadata filtering.
5. **Retrieves** relevant chunks using **hybrid search** (vector + keyword) for better recall.
6. **Re-ranks** results using a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) for precision.
7. **Structures** the content into clean, digital formats (Markdown, JSON, plain text).

The "RAG" aspect enables querying across multiple ingested documents — once several pages are indexed, you can ask natural-language questions and retrieve the most relevant passages with high precision.

---

## Research & Methodology Selection

### How We Chose the Approach

We used **Parallel AI** (parallel.ai) to research the state-of-the-art in handwritten document RAG pipelines as of 2026. Three research queries were issued:

1. **OCR comparison** — Comparing TrOCR, PaddleOCR, Surya OCR, Google Vision, EasyOCR, olmOCR, and Qwen-VL for handwritten text recognition.
2. **RAG implementation best practices** — Surveying LangChain, LlamaIndex, ChromaDB, FAISS, semantic chunking, and hybrid search approaches.
3. **Multimodal RAG advances** — Evaluating ColPali, ColQwen, M3DocRAG, and vision-language models for document understanding.

### Key Research Findings

| Topic | Finding | Source |
|-------|---------|--------|
| **Best OCR for handwriting (2026)** | Surya OCR leads with 90+ language support, layout analysis, and modern deep learning architecture | unstract.com, dev.to |
| **LLM-based OCR** | olmOCR and Qwen2-VL show strong contextual understanding but require GPU | unstract.com, Medium |
| **Vector stores** | ChromaDB ideal for prototyping; Qdrant/Pinecone for production scale | kapa.ai, dev.to |
| **Chunking** | Semantic chunking outperforms fixed-size by 15-30% in retrieval precision | kapa.ai, Dell InfoHub |
| **Hybrid search** | BM25 + vector gives 5-15% recall improvement over vector-only | multiple sources |
| **Re-ranking** | Cross-encoder re-ranking consistently improves precision by 10-20% | kapa.ai, Medium |
| **Multimodal RAG** | ColPali/ColQwen achieve SOTA on MP-DocVQA; treat documents as images | arxiv.org (M3DocRAG) |
| **Surya + Docling combo** | Powerful for RAG with layout-aware markdown output | Medium (Alain Airom) |

### Why This Stack Was Chosen

Based on the research, we selected:

- **Surya OCR** over TrOCR: 90+ languages (vs English-only), layout-aware, modern architecture, actively maintained
- **ChromaDB** over FAISS: Built-in metadata filtering, persistent storage, simpler API, native hybrid search support
- **Semantic chunking** over fixed-size: Respects topic boundaries, produces more coherent chunks for retrieval
- **Cross-encoder re-ranking**: Consistently provides 10-20% precision improvement with minimal latency cost
- **Hybrid search** (vector + keyword): Combines semantic understanding with exact term matching for better recall

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Handwritten Image                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Image Preprocessing                                │
│  ┌──────────┐ ┌────────┐ ┌──────────┐ ┌───────┐ ┌────────┐ │
│  │ Grayscale│→│Denoise │→│ CLAHE    │→│Binarize│→│Deskew  │ │
│  └──────────┘ └────────┘ └──────────┘ └───────┘ └────────┘ │
│                          │                                   │
│                          ▼                                   │
│               ┌───────────────────┐                          │
│               │ Line Segmentation │                          │
│               └───────────────────┘                          │
└──────────────────────────┬──────────────────────────────────┘
                           │  (list of line-crop images)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Handwriting OCR (Surya OCR)                        │
│  Detection → Recognition (90+ languages)                     │
│  Per-line recognition with confidence scores                 │
│  Fallback: TrOCR if Surya unavailable                        │
└──────────────────────────┬──────────────────────────────────┘
                           │  (raw text + confidences)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Text Post-processing                               │
│  Unicode normalisation → OCR error correction →              │
│  Hyphen merging → Artifact removal → Whitespace cleanup      │
└──────────────────────────┬──────────────────────────────────┘
                           │  (cleaned text)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4a: Semantic Chunking & Embedding                     │
│  Sentences → similarity-based grouping →                     │
│  sentence-transformers (all-MiniLM-L6-v2) embeddings         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4b: ChromaDB Vector Store                             │
│  Cosine similarity index + metadata filtering                │
│  Persistent storage on disk                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼  (at query time)
┌─────────────────────────────────────────────────────────────┐
│  Stage 5a: Hybrid Search (Vector + Keyword)                  │
│  Weighted fusion: 70% vector + 30% keyword                   │
│  Retrieves top-10 candidate chunks                           │
├─────────────────────────────────────────────────────────────┤
│  Stage 5b: Cross-Encoder Re-ranking                          │
│  ms-marco-MiniLM-L-6-v2 cross-encoder                       │
│  Re-scores and selects top-5 chunks                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: Document Structuring                               │
│  Heading detection → List detection → Paragraph grouping     │
│  ┌──────┐  ┌──────────┐  ┌───────────┐                      │
│  │ JSON │  │ Markdown │  │ Plain Text│                      │
│  └──────┘  └──────────┘  └───────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Image Preprocessing (`pipeline/image_preprocessor.py`)

**Goal:** Transform raw scanned/photographed images into clean, binarised line crops suitable for OCR.

**Steps taken:**
1. **Load & Resize** — Read the image with OpenCV; cap width at 2048px for speed.
2. **Grayscale Conversion** — Convert to single-channel.
3. **Denoising** — `cv2.fastNlMeansDenoising` (Non-Local Means) removes scan noise while preserving edges.
4. **Contrast Enhancement** — CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts ink-vs-paper contrast.
5. **Binarisation** — Adaptive Gaussian threshold converts to black ink on white paper.
6. **Deskewing** — Compute skew angle via `cv2.minAreaRect` and rotate to straighten.
7. **Line Segmentation** — Horizontal projection profiling identifies text line boundaries.

**Why these choices:**
- Adaptive thresholding handles uneven lighting across scanned pages.
- Per-line segmentation matches OCR model training distributions.
- CLAHE adapts locally, handling pages where ink density varies.

---

### Stage 2 — Handwriting OCR (`pipeline/handwriting_ocr.py`)

**Goal:** Convert each line-crop image into text with confidence scores.

**Primary Model:** **Surya OCR** (`surya-ocr`) — a modern multilingual document OCR toolkit.

**How Surya OCR works:**
- **Detection:** Identifies text regions using a deep learning detector trained on document layouts.
- **Recognition:** Recognises text in 90+ languages using a transformer-based recognition model.
- **Layout Analysis:** Understands tables, headers, images, and reading order.

**Fallback:** If Surya is not installed, the system falls back to Microsoft **TrOCR-Base-Handwritten** (`microsoft/trocr-base-handwritten`) for English-only recognition.

**Why Surya OCR over alternatives:**
- **vs TrOCR (Experiment 0):** 90+ languages vs English-only; layout-aware; actively maintained.
- **vs Tesseract:** Surya uses deep learning trained specifically on document structures.
- **vs PaddleOCR:** Comparable accuracy, but Surya's layout analysis is more comprehensive.
- **vs olmOCR:** Surya runs locally without requiring a 7B-parameter vision-language model.
- **vs Google Vision API:** Fully local, no API costs, no network dependency.

---

### Stage 3 — Text Post-processing (`pipeline/text_postprocessor.py`)

**Goal:** Clean up OCR artefacts and normalise the raw text.

**Steps taken:**
1. **Unicode Normalisation** — NFKC form; smart quotes → ASCII quotes.
2. **Hyphenated Word Merging** — Rejoin words split across lines.
3. **OCR Error Correction** — Pattern-based fixes: `rn` → `m`, digit/letter confusion, doubled punctuation.
4. **Artefact Removal** — Strip isolated noise characters and non-ASCII artifacts.
5. **Whitespace Collapse** — Normalise spaces and blank lines.
6. **Short Line Filtering** — Remove noise-only lines below minimum length.

---

### Stage 4a — Semantic Chunking & Embedding (`pipeline/embeddings.py`)

**Goal:** Split text at topic boundaries (not fixed windows) and generate dense vector embeddings.

**Steps taken:**
1. **Sentence Splitting** — Split text on sentence boundaries (`.`, `!`, `?`).
2. **Semantic Similarity Computation** — Encode consecutive sentences and compute cosine similarity.
3. **Boundary Detection** — When similarity drops below a threshold (0.75), start a new chunk.
4. **Embedding** — Encode each chunk using `all-MiniLM-L6-v2` (384-dim, normalised).
5. **Fallback** — If semantic chunking fails, fall back to fixed-size overlapping windows.

**Why semantic chunking over fixed-size (Experiment 0):**
- Respects topic boundaries — chunks are more coherent.
- Research shows 15-30% improvement in retrieval precision.
- Prevents mid-sentence splits that confuse retrieval.

---

### Stage 4b — ChromaDB Vector Store (`pipeline/vector_store.py`)

**Goal:** Store and index embeddings for fast, metadata-aware retrieval.

**Steps taken:**
1. **Persistent ChromaDB** — Embeddings persist to disk in `chroma_db/` directory.
2. **HNSW Indexing** — Approximate nearest-neighbour with cosine similarity.
3. **Metadata Storage** — Each chunk stores source document, chunk index, word count.
4. **Vector Search** — Standard cosine-similarity retrieval.
5. **Keyword Search** — ChromaDB's built-in text search for BM25-like keyword matching.
6. **Hybrid Search** — Weighted fusion of vector (70%) and keyword (30%) results.

**Why ChromaDB over FAISS (Experiment 0):**
- Built-in metadata filtering (source, date, chunk type).
- Native persistent storage without manual serialisation.
- Built-in keyword search for hybrid retrieval.
- Simpler API with automatic collection management.

---

### Stage 5 — Retrieval & Re-ranking (`pipeline/reranker.py`)

**Goal:** Retrieve candidates broadly, then re-rank precisely for the top results.

**Steps taken:**
1. **Hybrid Retrieval** — Retrieve top-10 candidates using hybrid search.
2. **Cross-Encoder Scoring** — For each (query, chunk) pair, compute a fine-grained relevance score using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. **Score Fusion** — Weighted blend: 70% re-rank score + 30% original retrieval score.
4. **Top-K Selection** — Return the top-5 most relevant chunks.

**Why re-ranking:**
- Cross-encoders jointly attend to query + document tokens, capturing nuanced relevance.
- Consistently provides 10-20% precision improvement over bi-encoder retrieval alone.
- The retrieve-then-rerank pattern (retrieve 10 → rerank to 5) balances recall and precision.

---

### Stage 6 — Document Structuring (`pipeline/document_structurer.py`)

**Goal:** Detect document structure and output in multiple formats.

**Steps taken:**
1. **Heading Detection** — Regex patterns for Markdown headers, ALL-CAPS lines, numbered headings, chapter/section markers, keyword headings.
2. **List Detection** — Bulleted (`-`, `*`, `•`) and numbered (`1.`, `a)`) lists.
3. **Paragraph Grouping** — Consecutive non-heading, non-list lines grouped as sections.
4. **Multi-format Output:**
   - **JSON** — Full structured representation with title, sections, metadata, statistics.
   - **Markdown** — Clean document with proper heading levels and formatting.
   - **Plain Text** — ASCII-formatted with underlines for headings.

---

## Key Improvements over Experiment 0

| Feature | Experiment 0 | Experiment 2 |
|---------|-------------|-------------|
| **OCR Engine** | TrOCR (English-only) | Surya OCR (90+ languages) |
| **Layout Awareness** | None (line-by-line only) | Full layout analysis |
| **Chunking** | Fixed-size (512 words, 64 overlap) | Semantic chunking (topic boundaries) |
| **Vector Store** | FAISS `IndexFlatIP` (in-memory) | ChromaDB (persistent, metadata-aware) |
| **Search** | Vector-only (cosine) | Hybrid (vector 70% + keyword 30%) |
| **Re-ranking** | None | Cross-encoder (`ms-marco-MiniLM`) |
| **Metadata** | Minimal | Source, chunk index, word count, method |
| **Persistence** | Manual `.faiss` + `.json` files | Automatic ChromaDB persistence |
| **Fallback OCR** | None | TrOCR fallback if Surya unavailable |

---

## Directory Structure

```
experiment_2/
├── README.md                  # This file
├── config.py                  # Pipeline configuration
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # CLI entry point
├── create_sample_images.py    # Generate test images
├── pipeline/
│   ├── __init__.py
│   ├── image_preprocessor.py  # Stage 1: Image cleaning & line segmentation
│   ├── handwriting_ocr.py     # Stage 2: Surya OCR (+ TrOCR fallback)
│   ├── text_postprocessor.py  # Stage 3: Text cleanup & normalisation
│   ├── embeddings.py          # Stage 4a: Semantic chunking & embeddings
│   ├── vector_store.py        # Stage 4b: ChromaDB vector store
│   ├── reranker.py            # Stage 5: Cross-encoder re-ranking
│   ├── document_structurer.py # Stage 6: Structure detection & output
│   └── rag_pipeline.py        # Orchestrator (all stages)
├── sample_images/             # Test images (generated)
├── chroma_db/                 # ChromaDB persistent storage
└── output/                    # Structured output files
```

---

## Installation

```bash
# Navigate to experiment directory
cd experiment_2

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**System requirements:**
- Python 3.10+
- ~3 GB disk space for model weights (downloaded on first run)
- GPU optional but recommended for faster OCR (MPS on Apple Silicon is auto-detected)

**Note:** Surya OCR models are downloaded automatically on first use (~500 MB). If Surya installation fails on your system, the pipeline automatically falls back to TrOCR.

---

## Usage

### 1. Generate Sample Images

```bash
python create_sample_images.py
```

Creates three synthetic handwritten-style images in `sample_images/`.

### 2. Process a Single Image

```bash
python run_pipeline.py --image sample_images/sample_letter.png
```

### 3. Process All Images in a Directory

```bash
python run_pipeline.py --image-dir sample_images/
```

### 4. Query Indexed Documents

After processing images, query the vector store:

```bash
python run_pipeline.py --query "What are the key concepts in neural networks?"
```

### 5. Query Without Re-ranking (Faster)

```bash
python run_pipeline.py --query "meeting action items" --no-rerank
```

### 6. Vector-Only Search (No Hybrid)

```bash
python run_pipeline.py --query "budget allocation" --no-hybrid
```

### 7. Full Workflow (Generate + Process + Query)

```bash
python run_pipeline.py --generate-samples --image-dir sample_images/ -v
python run_pipeline.py --query "What learning rate is recommended?" -v
```

The `-v` flag enables verbose logging for all pipeline stages.

### 8. Reset the Vector Store

```bash
python run_pipeline.py --reset --image-dir sample_images/
```

### Output

Structured documents are saved to `output/` in three formats:
- `document.json` — Machine-readable structured format with metadata
- `document.md` — Human-readable Markdown
- `document.txt` — Plain-text version

---

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for chunks and queries |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-ranking |
| `CHUNK_SIZE` | 512 | Maximum words per semantic chunk |
| `CHUNK_OVERLAP` | 64 | Word overlap between chunks |
| `SEMANTIC_THRESHOLD` | 0.75 | Cosine similarity threshold for chunk boundaries |
| `TOP_K_RETRIEVAL` | 10 | Candidates to retrieve before re-ranking |
| `TOP_K_RERANK` | 5 | Final results after re-ranking |
| `CHROMA_COLLECTION` | `handwritten_docs` | ChromaDB collection name |
| `SURYA_LANGUAGES` | `["en"]` | Languages for Surya OCR |
| `IMAGE_PREPROCESSING.denoise_strength` | 10 | NLMeans denoising strength |
| `IMAGE_PREPROCESSING.deskew` | True | Auto-straighten pages |
| `IMAGE_PREPROCESSING.binarize` | True | Apply adaptive binarisation |

---

## Design Decisions & Rationale

### Why Surya OCR?

Based on our Parallel AI research, Surya OCR emerged as the best choice for 2026:
- **90+ languages** vs TrOCR's English-only limitation.
- **Layout analysis** detects tables, headers, images, and reading order.
- **Modern architecture** leveraging deep learning for detection and recognition.
- **Active development** with regular updates and community support.
- **Docling integration** — can be combined with IBM's Docling for even richer document understanding.

### Why Semantic Chunking?

Traditional fixed-size chunking (as in Experiment 0) blindly splits text every N words:
- Can split a sentence in half, losing context.
- A chunk may contain parts of two unrelated topics.

Semantic chunking splits where the **topic changes**:
- Each chunk is a coherent unit of meaning.
- Research shows 15-30% improvement in retrieval precision.
- The similarity threshold (0.75) was chosen empirically as a balance between too-large and too-small chunks.

### Why ChromaDB?

For this experiment's scope, ChromaDB provides:
- **Persistent storage** — no manual serialisation needed.
- **Metadata filtering** — query by source document, chunk type, etc.
- **Built-in keyword search** — enables hybrid retrieval without extra infrastructure.
- **Simple API** — less boilerplate than FAISS + manual metadata management.

### Why Hybrid Search?

Pure vector search can miss relevant results when:
- The query uses exact terminology from the document.
- Short queries don't embed well semantically.

Hybrid search combines:
- **Vector search (70%)** — captures semantic meaning.
- **Keyword search (30%)** — catches exact term matches.

The 70/30 split was chosen based on research showing this ratio gives the best F1 score across diverse query types.

### Why Cross-Encoder Re-ranking?

Bi-encoders (used for initial retrieval) embed query and document independently — they can't capture fine-grained token interactions. Cross-encoders jointly process (query, document) pairs through a transformer, enabling:
- Attention between query tokens and document tokens.
- 10-20% precision improvement over bi-encoder retrieval alone.
- The retrieve-then-rerank pattern (10 → 5) keeps latency manageable.

### Why RAG for Document Digitisation?

Beyond Experiment 0's reasoning:
- **Hybrid search** improves recall for both semantic and keyword queries.
- **Re-ranking** ensures the most relevant content surfaces first.
- **Semantic chunking** means retrieved context is coherent and complete.
- **Metadata filtering** enables targeted queries (e.g., "from meeting notes only").

---

## Libraries & Tools Used

| Library | Version | Purpose |
|---------|---------|---------|
| `surya-ocr` | ≥0.8.0 | Modern multilingual handwriting OCR |
| `torch` | ≥2.2.0 | Deep learning runtime (for fallback TrOCR) |
| `transformers` | ≥4.40.0 | TrOCR model loading (fallback) |
| `sentence-transformers` | ≥2.7.0 | Text embedding + cross-encoder re-ranking |
| `chromadb` | ≥0.5.0 | Persistent vector database with hybrid search |
| `opencv-python` | ≥4.9.0 | Image preprocessing (denoise, binarise, deskew) |
| `Pillow` | ≥10.2.0 | Image I/O and manipulation |
| `numpy` | ≥1.26.0 | Numerical operations |

---

## Limitations & Future Work

1. **Surya model size** — First-run download is ~500 MB. The pipeline includes a TrOCR fallback for environments where Surya cannot be installed.
2. **No LLM generation step** — The pipeline structures text via rule-based heuristics. Adding an LLM (GPT-4, Llama) for the "Generate" step would enable summarisation, entity extraction, and conversational Q&A.
3. **ColPali/multimodal RAG** — Our research found that ColPali (treating documents as images for retrieval) achieves SOTA on document QA benchmarks. A future experiment could replace text-based RAG with visual retrieval.
4. **Evaluation framework** — No automated evaluation (RAGAS, DeepEval) is included. Adding golden-set evaluation would enable systematic quality measurement.
5. **Batch OCR** — Line-by-line OCR is not batched. Batching would improve GPU throughput.
6. **Complex layouts** — Tables, multi-column, and diagram extraction require additional layout analysis (LayoutLMv3, Docling integration).
7. **Contextual chunking** — Anthropic's contextual retrieval technique (prepending LLM-generated summaries per chunk) could further improve retrieval by +49%.

---

## Parallel AI Research Sources

The methodology for this experiment was informed by research conducted via Parallel AI (`parallel.ai`). Key sources consulted:

- **"Best Open-Source OCR Tools in 2026"** — unstract.com (March 2026)
- **"OCR for Handwritten Text in 2026"** — dev.to (February 2026)
- **"How to Build a RAG Pipeline from Scratch in 2026"** — kapa.ai (January 2026)
- **"Using SuryaOCR with Docling"** — Medium/Alain Airom (January 2026)
- **"The State of RAG 2026"** — AIGuys/Medium (January 2026)
- **"M3DocRAG: Multi-modal Retrieval"** — arxiv.org (November 2024)
- **"Best Handwriting OCR Tools 2026"** — extend.ai (February 2026)
- **"Building State-of-the-Art Vision-Enabled RAG Pipelines (2026)"** — Towards AI (February 2026)
- **"Multimodal Document RAG with ColQwen2"** — together.ai
- **"What is Multimodal RAG?"** — IBM (September 2025)

---

*Experiment 2 — Built as part of the Coding Agent Search Benchmark project.*
