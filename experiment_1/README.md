# Experiment 1 — Advanced Handwritten Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that takes **handwritten document images** as input, extracts text via TrOCR, and produces **structured digital documents** (Markdown, JSON, plain text) as output — enhanced with **hybrid search**, **cross-encoder re-ranking**, and **ChromaDB** persistence.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences from Experiment 0](#key-differences-from-experiment-0)
3. [Architecture](#architecture)
4. [Pipeline Stages — Step-by-Step Development](#pipeline-stages--step-by-step-development)
5. [Directory Structure](#directory-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Design Decisions & Methodology](#design-decisions--methodology)
10. [Libraries & Tools Used](#libraries--tools-used)
11. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This experiment builds an end-to-end RAG pipeline for digitising handwritten documents with significantly enhanced retrieval capabilities compared to Experiment 0. The pipeline:

1. **Ingests** scanned or photographed handwritten document images.
2. **Preprocesses** images with morphological cleaning, adaptive denoising, contrast enhancement, and line segmentation.
3. **Extracts** text using Microsoft TrOCR with optional batch inference for GPU acceleration.
4. **Post-processes** OCR output with extended error correction, sentence boundary repair, and artefact removal.
5. **Chunks** text using a recursive character text splitter (paragraph → sentence → word hierarchy) with metadata enrichment.
6. **Indexes** chunks in a ChromaDB persistent vector store with cosine similarity.
7. **Retrieves** relevant content using hybrid BM25 + vector search fused via Reciprocal Rank Fusion (RRF).
8. **Re-ranks** candidates with a cross-encoder model for higher precision.
9. **Structures** the retrieved content into clean Markdown, JSON, and plain-text documents.

---

## Key Differences from Experiment 0

| Feature | Experiment 0 | Experiment 1 |
|---------|-------------|-------------|
| **Vector Store** | FAISS (IndexFlatIP) | ChromaDB (persistent, metadata filtering) |
| **Retrieval** | Vector-only | Hybrid: BM25 + vector with RRF |
| **Re-ranking** | None | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Chunking** | Simple word-level splitting | Recursive character splitting (paragraph → sentence → word) |
| **Chunk Metadata** | Basic (source, chunk_id) | Rich (source, index, char/word counts, OCR confidence, timestamp) |
| **Image Preprocessing** | Standard pipeline | + Morphological opening/closing for noise removal |
| **OCR Inference** | Sequential line-by-line | Batch inference support for GPU |
| **Text Post-processing** | Basic patterns | + Sentence boundary repair, extended substitutions |
| **Document Structuring** | Headings/lists | + Salutation detection, date extraction, section type classification |
| **Dependencies** | faiss-cpu | chromadb, rank-bm25 (no faiss) |

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
│  Grayscale → Denoise → Contrast (CLAHE) → Binarize →        │
│  Morphological Cleanup (open/close) → Deskew →               │
│  Line Segmentation                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │  (list of line-crop images)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Handwriting OCR (TrOCR)                            │
│  ViT Encoder → Text Transformer Decoder                      │
│  Batch inference on GPU, per-line confidence scores          │
└──────────────────────────┬──────────────────────────────────┘
                           │  (raw text + confidences)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Text Post-processing                               │
│  Unicode normalise → OCR error correction → Hyphen merge →   │
│  Sentence boundary repair → Artefact removal → Whitespace    │
└──────────────────────────┬──────────────────────────────────┘
                           │  (cleaned text)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Recursive Chunking + Metadata                      │
│  Paragraph → Sentence → Word splitting hierarchy             │
│  + source, index, word/char counts, OCR confidence, timestamp│
└──────────────────────────┬──────────────────────────────────┘
                           │  (enriched Chunk objects)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Embed & Index in ChromaDB                          │
│  sentence-transformers → normalised vectors →                │
│  ChromaDB persistent collection (cosine)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │  (indexed vectors + metadata)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: Hybrid Retrieval + Re-ranking                      │
│  ┌──────────┐  ┌──────────┐                                  │
│  │  Vector   │  │  BM25    │                                  │
│  │  Search   │  │  Search  │                                  │
│  └─────┬────┘  └─────┬────┘                                  │
│        └──────┬──────┘                                        │
│               ▼                                               │
│     Reciprocal Rank Fusion (RRF)                              │
│               │                                               │
│               ▼                                               │
│     Cross-Encoder Re-ranking                                  │
│     (ms-marco-MiniLM-L-6-v2)                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │  (top-k re-ranked chunks)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 7: Document Structuring                               │
│  Heading detection → List detection → Salutation detection → │
│  Date extraction → Section type classification               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 8: Output Generation                                  │
│  ┌──────┐  ┌──────────┐  ┌───────────┐                      │
│  │ JSON │  │ Markdown │  │ Plain Text│                      │
│  └──────┘  └──────────┘  └───────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages — Step-by-Step Development

### Stage 1 — Image Preprocessing (`pipeline/image_preprocessor.py`)

**Goal:** Transform raw scanned/photographed images into clean, binarised line crops suitable for OCR.

**Steps taken:**

1. **Load & Resize** — Read the image with OpenCV and cap the width at 2048px to keep processing fast without sacrificing detail.
2. **Grayscale Conversion** — Convert BGR to single-channel grayscale.
3. **Denoising** — Apply `cv2.fastNlMeansDenoising` (Non-Local Means algorithm) to remove scan noise and camera grain while preserving edges.
4. **Contrast Enhancement** — Use CLAHE (Contrast Limited Adaptive Histogram Equalisation) with a 2.0 clip limit and 8×8 tile grid to boost ink-vs-paper contrast for faded handwriting.
5. **Binarisation** — Adaptive Gaussian thresholding (block size 15, constant 11) converts the image to black ink on white paper. Adaptive thresholding handles uneven lighting across the page.
6. **Morphological Cleanup** *(new in Experiment 1)* — Two-stage morphological operation:
   - **Opening** (2×2 elliptical kernel) removes small noise blobs.
   - **Closing** (3×3 elliptical kernel) fills tiny gaps in ink strokes.
   This replaces manual noise thresholding and produces cleaner line segmentation input.
7. **Deskewing** — Compute the skew angle via `cv2.minAreaRect` on ink pixels and rotate if angle exceeds 0.5°.
8. **Line Segmentation** — Horizontal projection profiling with configurable minimum line height and adaptive padding (25% of minimum height). Each line is cropped individually for per-line OCR.

**Why morphological cleanup:**
- Removes salt-and-pepper noise that causes spurious line detections.
- Fills small gaps in pen strokes that could cause TrOCR to mis-segment characters.
- Two small kernels avoid destroying fine handwriting details.

---

### Stage 2 — Handwriting OCR (`pipeline/handwriting_ocr.py`)

**Goal:** Convert each line-crop image into text with confidence scores.

**Model:** Microsoft **TrOCR-Large-Handwritten** (`microsoft/trocr-large-handwritten`)

**Steps taken:**

1. Auto-detect device (CUDA → MPS → CPU).
2. Lazy-load the TrOCR processor and model on first call.
3. **Batch inference** *(new in Experiment 1)*: When a GPU is available, process multiple line images simultaneously (configurable batch_size=4). Falls back to sequential processing on CPU or if batch inference fails.
4. For each line image, generate text tokens with `model.generate()`, compute per-token softmax probabilities, and average them into a line-level confidence score.
5. Each line result includes word count in addition to text and confidence.
6. Aggregate all line results into a `DocumentOCRResult` with overall statistics (average/min/max confidence, total word count).

**Why batch inference:**
- 2-4x faster on GPU for multi-line documents.
- Graceful fallback to sequential ensures robustness on CPU.

---

### Stage 3 — Text Post-processing (`pipeline/text_postprocessor.py`)

**Goal:** Clean up OCR artefacts and normalise the raw text.

**Steps taken:**

1. **Unicode Normalisation** — NFKC normalisation; smart quotes → ASCII; non-breaking spaces → regular spaces.
2. **Hyphenated Word Merging** — Rejoin words split across lines by hyphens.
3. **Extended OCR Error Correction** *(expanded in Experiment 1)* — Pattern-based fixes for:
   - `0` ↔ `o`, `1` ↔ `l` (classic confusions)
   - `rn` → `m` (ligature error)
   - `vv` → `w` (character split)
   - `cI` → `d` (misread)
   - Doubled punctuation collapse
4. **Artefact Removal** — Strip isolated single-character noise tokens.
5. **Sentence Boundary Repair** *(new in Experiment 1)* — Capitalise the first letter after sentence-ending punctuation (`. `, `! `, `? `), correcting OCR-induced case errors.
6. **Whitespace Collapse** — Normalise runs of spaces and blank lines.

**Why sentence boundary repair:**
- OCR often miscases letters after punctuation. This heuristic restores proper sentence capitalisation without needing a full language model.

---

### Stage 4 — Recursive Chunking (`pipeline/chunker.py`)

**Goal:** Break text into searchable chunks using a natural-boundary-first approach.

**Steps taken:**

1. **Recursive character splitting** *(new in Experiment 1)* — Instead of fixed word-count windows, the chunker tries separators in order:
   - `\n\n` (paragraph breaks)
   - `\n` (line breaks)
   - `. ` (sentence boundaries)
   - ` ` (word boundaries)
   - `""` (character-level fallback)
2. For each separator level, splits that fit within `chunk_size` (512 chars) are merged together. Splits exceeding the limit recurse to the next separator level.
3. **Overlap preservation** — When merging, trailing parts of the previous chunk are retained up to `chunk_overlap` (64 chars) to maintain context across boundaries.
4. **Metadata enrichment** — Each `Chunk` object includes:
   - `source`: origin image path
   - `chunk_index`: position in document
   - `char_count`, `word_count`: size metrics
   - `created_at`: UTC ISO timestamp
   - `ocr_confidence`: from the OCR stage

**Why recursive splitting over word-level:**
- Preserves paragraph and sentence integrity when possible.
- Falls back gracefully to smaller units only for oversized blocks.
- Rich metadata enables downstream filtering (e.g., retrieve only high-confidence chunks).

---

### Stage 5 — Embed & Index in ChromaDB (`pipeline/embeddings.py`, `pipeline/vector_store.py`)

**Goal:** Store chunk embeddings in a persistent vector database with metadata.

**Steps taken:**

1. **Embedding** — Encode each chunk using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) with L2 normalisation enabled at the model level.
2. **ChromaDB Indexing** *(replacing FAISS from Experiment 0)*:
   - Create a persistent client at the configured directory.
   - Create/get a collection named `handwritten_docs` with HNSW cosine space.
   - Add embeddings, documents, and metadata in a single call.
   - ChromaDB handles index persistence, compaction, and metadata filtering automatically.
3. **All-documents access** — The store exposes `get_all_documents()` for BM25 index construction.

**Why ChromaDB over FAISS:**
- **Persistence** is built-in — no manual serialisation of index files and metadata JSON.
- **Metadata filtering** — can filter by source, confidence, or timestamp at query time.
- **Collection management** — supports multiple collections, reset, and count without custom code.
- **HNSW index** — approximate nearest neighbours that scale better than flat search for larger collections.

**Why all-MiniLM-L6-v2:**
- Same reasoning as Experiment 0: best speed/quality trade-off at 384 dimensions on CPU. Chosen to keep the embedding comparison fair across experiments.

---

### Stage 6 — Hybrid Retrieval + Re-ranking (`pipeline/retriever.py`)

**Goal:** Combine keyword and semantic search for more robust retrieval, then refine with pairwise relevance scoring.

This is the **primary enhancement** of Experiment 1.

**Steps taken:**

1. **Vector Search** — Query ChromaDB for the top 20 candidates by cosine similarity.
2. **BM25 Search** — Build a BM25Okapi index over all stored document texts (tokenised by whitespace + lowercased). Retrieve top 20 candidates by keyword relevance.
3. **Reciprocal Rank Fusion (RRF)** — Merge the two ranked lists without score normalisation:
   - For each result at rank `r` in the vector list: `rrf_score += alpha / (k + r + 1)`
   - For each result at rank `r` in the BM25 list: `rrf_score += (1 - alpha) / (k + r + 1)`
   - Default: `alpha=0.7` (70% weight to vector), `k=60` (standard RRF constant).
   - Sort by combined RRF score.
4. **Cross-Encoder Re-ranking** — Pass the top fused candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2`, which scores each (query, document) pair with a single forward pass. Select the top 5 by re-ranker score.

**Why hybrid search:**
- BM25 excels at exact keyword matches (names, numbers, technical terms) that embedding models can miss.
- Vector search captures semantic paraphrases and conceptual similarity.
- Research shows hybrid search consistently delivers 5-15% recall improvement over either method alone (referenced in skill.md).

**Why RRF over weighted score averaging:**
- RRF is score-distribution agnostic — it works without normalising BM25 and cosine scores to the same scale.
- Only requires rank positions, making it robust to score magnitude differences.

**Why cross-encoder re-ranking:**
- Two-stage retrieval (retrieve broadly, re-rank precisely) is a well-established pattern in information retrieval.
- Cross-encoders jointly attend to query and document tokens, producing more accurate relevance judgments than bi-encoder similarity.
- `ms-marco-MiniLM-L-6-v2` is small (22M parameters) and fast enough for real-time re-ranking of 20 candidates.

---

### Stage 7 — Document Structuring (`pipeline/document_structurer.py`)

**Goal:** Detect document structure and output in multiple formats.

**Steps taken:**

1. **Heading Detection** — Regex patterns recognise:
   - Markdown-style headers (`#`, `##`, etc.)
   - ALL-CAPS lines
   - Numbered headings (`1. Topic`)
   - Chapter/Section/Part markers
2. **List Detection** — Identifies bulleted (`-`, `•`, `*`) and numbered (`1.`, `a)`) lists.
3. **Salutation Detection** *(new in Experiment 1)* — Recognises letter openings (Dear, Hello) and closings (Sincerely, Regards). These are classified as `salutation` section types.
4. **Date Extraction** *(new in Experiment 1)* — Regex extracts dates in common formats (MM/DD/YYYY, Month DD, YYYY) and adds them to document metadata.
5. **Section Type Classification** — Each section is classified as `paragraph`, `list`, or `salutation` based on content analysis.
6. **Multi-format Output:**
   - **JSON** — Full structured representation with title, sections (including section types), metadata.
   - **Markdown** — Clean, readable document with proper heading levels.
   - **Plain Text** — ASCII-formatted version with underlines for headings.

---

### Stage 8 — Pipeline Orchestration (`pipeline/rag_pipeline.py`)

**Goal:** Tie all components together into a single, configurable pipeline.

**Steps taken:**

1. The `HandwritingRAGPipeline` class initialises all sub-components (preprocessor, OCR, postprocessor, chunker, embedding engine, vector store, hybrid retriever, structurer).
2. `process_image()` runs all 8 stages sequentially, capturing timing data for each.
3. `query()` runs hybrid search + re-ranking across all indexed documents.
4. `query_simple()` provides a vector-only search fallback.
5. `process_batch()` handles multi-image ingestion.
6. All parameters are configurable via `config.py`.

---

## Directory Structure

```
experiment_1/
├── README.md                  # This file
├── config.py                  # Pipeline configuration
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # CLI entry point
├── create_sample_images.py    # Generate test images
├── skill.md                   # RAG skill reference
├── ingestion-and-chunking.md  # Reference: chunking strategies
├── embeddings-and-retrieval.md# Reference: embeddings & retrieval
├── advanced-rag.md            # Reference: advanced RAG techniques
├── evaluation.md              # Reference: RAG evaluation
├── pipeline/
│   ├── __init__.py
│   ├── image_preprocessor.py  # Stage 1: Image cleaning + morphological ops
│   ├── handwriting_ocr.py     # Stage 2: TrOCR recognition + batch inference
│   ├── text_postprocessor.py  # Stage 3: Text cleanup + sentence repair
│   ├── chunker.py             # Stage 4: Recursive splitting + metadata
│   ├── embeddings.py          # Stage 5a: Sentence embeddings
│   ├── vector_store.py        # Stage 5b: ChromaDB index
│   ├── retriever.py           # Stage 6: Hybrid BM25+vector + re-ranking
│   ├── document_structurer.py # Stage 7: Structure detection
│   └── rag_pipeline.py        # Orchestrator (all stages)
├── sample_images/             # Test images (generated)
├── chroma_db/                 # ChromaDB persistent store
└── output/                    # Structured output files
```

---

## Installation

```bash
# Navigate to experiment directory
cd experiment_1

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**System requirements:**
- Python 3.10+
- ~2 GB disk space for model weights (downloaded on first run)
- ~100 MB for cross-encoder re-ranker model
- GPU optional but recommended for faster OCR (MPS on Apple Silicon is auto-detected)

---

## Usage

### 1. Generate Sample Images

```bash
python create_sample_images.py
```

Creates four synthetic handwritten-style images in `sample_images/` (letter, notes, recipe, meeting notes).

### 2. Process a Single Image

```bash
python run_pipeline.py --image sample_images/sample_letter.png
```

### 3. Process All Images in a Directory

```bash
python run_pipeline.py --image-dir sample_images/
```

### 4. Query Indexed Documents (Hybrid + Re-ranking)

After processing images, query the vector store with full hybrid search:

```bash
python run_pipeline.py --query "What are the key concepts in neural networks?"
```

### 5. Query with Vector-Only Search

Skip BM25 and re-ranking for a faster, simpler search:

```bash
python run_pipeline.py --query "chocolate cake ingredients" --simple
```

### 6. Full Workflow (Generate + Process)

```bash
python run_pipeline.py --generate-samples --image-dir sample_images/ -v
```

The `-v` flag enables verbose logging for all pipeline stages.

### Output

Structured documents are saved to `output/` in three formats:
- `document.json` — Machine-readable structured format with section types and metadata
- `document.md` — Human-readable Markdown
- `document.txt` — Plain-text version

---

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TROCR_MODEL` | `microsoft/trocr-large-handwritten` | HuggingFace model for OCR |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-ranking |
| `CHUNK_SIZE` | 512 | Characters per chunk (recursive splitting) |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `CHUNK_SEPARATORS` | `["\n\n", "\n", ". ", " ", ""]` | Separator hierarchy |
| `TOP_K_RETRIEVAL` | 5 | Final number of results returned |
| `TOP_K_RERANK_CANDIDATES` | 20 | Candidates fetched before re-ranking |
| `HYBRID_ALPHA` | 0.7 | Vector weight in RRF (0.3 for BM25) |
| `RRF_K` | 60 | Standard RRF constant |
| `IMAGE_PREPROCESSING.morphological_clean` | True | Enable morphological noise removal |
| `IMAGE_PREPROCESSING.denoise_strength` | 10 | NLMeans denoising strength |

---

## Design Decisions & Methodology

### Why ChromaDB over FAISS?

Experiment 0 used FAISS `IndexFlatIP` with manual JSON metadata persistence. ChromaDB was chosen for Experiment 1 because:
- **Native persistence**: No manual save/load of index files and metadata.
- **Metadata filtering**: Filter results by source document, confidence, or timestamp at query time — impossible with bare FAISS.
- **Collection management**: Multiple collections, reset, count — without custom wrapper code.
- **HNSW index**: Approximate nearest neighbours that scale sublinearly, vs FAISS flat search's O(n).
- **Trade-off**: ChromaDB has higher overhead for very small collections (<100 chunks). At document-digitisation scale (hundreds to thousands of chunks), the ergonomic advantages outweigh the performance cost.

### Why Hybrid Search?

Embedding-based search alone has a known weakness: it can miss exact keyword matches. A query like "350 degrees" might not retrieve a chunk containing exactly "350 degrees F" if the embedding space doesn't strongly encode numerical tokens. BM25 handles this trivially via exact token matching. The skill reference (Experiment 1 skill.md) explicitly recommends hybrid search as the "gold standard for retrieval" with 5-15% recall improvement.

### Why Reciprocal Rank Fusion?

Two alternative fusion methods exist:
- **Weighted score averaging**: Requires normalising BM25 scores (unbounded) and cosine scores ([-1, 1]) to the same scale — fragile and data-dependent.
- **CombMNZ**: Rewards documents appearing in multiple lists but still needs score normalisation.
- **RRF**: Only uses rank positions, making it completely score-agnostic. The formula `1/(k + rank)` with k=60 is a well-studied default (Cormack et al., 2009).

### Why Cross-Encoder Re-ranking?

Bi-encoder embeddings (used for initial retrieval) encode query and document independently — they cannot model fine-grained token interactions. Cross-encoders process the (query, document) pair jointly through all transformer layers, capturing nuanced relevance signals. The skill reference recommends "retrieve 20, re-rank to top 5" as a pattern that "consistently improves precision."

### Why Recursive Character Splitting?

Experiment 0's word-level windowing splits mid-sentence without regard for natural boundaries. Recursive splitting preserves paragraph and sentence structure when chunks fit within the size budget, only resorting to word-level or character-level splitting when necessary. This produces more coherent chunks that improve both embedding quality and downstream structuring.

### Why Per-line OCR Instead of Full-page?

Same rationale as Experiment 0: TrOCR was trained on single text lines from the IAM dataset. Feeding full pages degrades accuracy significantly.

---

## Libraries & Tools Used

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.2.0 | Deep learning runtime |
| `transformers` | ≥4.40.0 | TrOCR model loading & inference |
| `sentence-transformers` | ≥2.7.0 | Text embedding + cross-encoder re-ranking |
| `opencv-python` | ≥4.9.0 | Image preprocessing (denoise, binarise, morphological ops, deskew) |
| `Pillow` | ≥10.2.0 | Image I/O and manipulation |
| `chromadb` | ≥0.5.0 | Persistent vector store with HNSW indexing |
| `rank-bm25` | ≥0.2.2 | BM25Okapi keyword search for hybrid retrieval |
| `numpy` | ≥1.26.0 | Numerical operations |

---

## Limitations & Future Work

1. **English-only** — TrOCR-Large-Handwritten is trained on English. Multilingual support would require a multilingual model or fine-tuning.
2. **No LLM generation step** — The current pipeline structures text via rule-based heuristics. Adding an LLM for the "Generate" step would enable summarisation, entity extraction, and conversational Q&A.
3. **BM25 index rebuilding** — The BM25 index is rebuilt from scratch on each query. For production use, it should be cached and incrementally updated.
4. **Cross-encoder latency** — Re-ranking 20 candidates adds ~200-500ms on CPU. For latency-critical applications, a faster re-ranker (e.g., ColBERT) could be substituted.
5. **Layout analysis** — Complex layouts (tables, multi-column, diagrams) are not handled. Integrating LayoutLMv3 or Donut is a natural extension.
6. **Evaluation metrics** — No automated evaluation (RAGAS, DeepEval) is integrated yet. The evaluation reference files provide a roadmap for adding this.
7. **Confidence-based rejection** — Low-confidence OCR lines could be flagged for human review rather than silently included.

---

*Experiment 1 — Built as part of the Coding Agent Search Benchmark project.*
