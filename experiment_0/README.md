# Experiment 0 — Handwritten Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that takes **handwritten document images** as input, extracts text via state-of-the-art OCR, and produces **structured digital documents** (Markdown, JSON, plain text) as output.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Directory Structure](#directory-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Design Decisions & Methodology](#design-decisions--methodology)
9. [Libraries & Tools Used](#libraries--tools-used)
10. [Limitations & Future Work](#limitations--future-work)

---

## Overview

Handwritten documents — letters, notes, recipes, meeting minutes — are ubiquitous but unsearchable. This experiment builds an end-to-end pipeline that:

1. **Ingests** scanned or photographed handwritten document images.
2. **Extracts** text using Microsoft TrOCR, a Vision-Transformer + Text-Transformer model fine-tuned on handwritten text.
3. **Indexes** extracted text into a FAISS vector store for semantic retrieval.
4. **Structures** the content into clean, digital formats (Markdown, JSON, plain text) with headings, paragraphs, and lists detected automatically.

The "RAG" aspect enables querying across multiple ingested documents — once several pages are indexed, you can ask natural-language questions and retrieve the most relevant passages.

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
│  │ Grayscale│→│Denoise │→│ Contrast │→│Binarize│→│Deskew  │ │
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
│  Stage 2: Handwriting OCR (TrOCR)                            │
│  ViT Encoder → Text Transformer Decoder                      │
│  Per-line recognition with confidence scores                 │
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
│  Stage 4: Chunking & Embedding                               │
│  Text → overlapping chunks → sentence-transformers →         │
│  FAISS IndexFlatIP (cosine similarity)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  (indexed vectors)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Retrieval-Augmented Structuring                    │
│  Query vector store → retrieve top-k chunks →                │
│  Detect headings / paragraphs / lists →                      │
│  Produce StructuredDocument                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: Output Generation                                  │
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
1. **Load & Resize** — Read the image with OpenCV and cap the width at 2048px to keep processing fast without sacrificing detail.
2. **Grayscale Conversion** — Convert BGR to single-channel grayscale.
3. **Denoising** — Apply `cv2.fastNlMeansDenoising` (Non-Local Means algorithm) to remove scan noise and camera grain while preserving edges.
4. **Contrast Enhancement** — Use CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost ink-vs-paper contrast, critical for faded handwriting.
5. **Binarisation** — Adaptive Gaussian thresholding converts the image to black ink on white paper. Adaptive thresholding handles uneven lighting across the page.
6. **Deskewing** — Compute the skew angle via `cv2.minAreaRect` on ink pixels and rotate the image to straighten text lines.
7. **Line Segmentation** — Horizontal projection profiling (summing pixel intensities per row) identifies gaps between text lines. Each line is cropped individually for per-line OCR.

**Why these choices:**
- Adaptive thresholding outperforms global thresholding on real scans with shadows or creases.
- Per-line segmentation dramatically improves TrOCR accuracy, which expects single-line inputs.
- CLAHE adapts locally, handling pages where ink density varies.

---

### Stage 2 — Handwriting OCR (`pipeline/handwriting_ocr.py`)

**Goal:** Convert each line-crop image into text.

**Model:** Microsoft **TrOCR-Large-Handwritten** (`microsoft/trocr-large-handwritten`)

**How it works:**
- **Encoder:** A Vision Transformer (ViT) processes the line image into a sequence of patch embeddings.
- **Decoder:** A text Transformer (initialised from RoBERTa) auto-regressively generates the recognised text.
- The model was pre-trained on synthetic handwriting data and fine-tuned on IAM Handwriting Dataset — the largest publicly available English handwriting corpus.

**Steps taken:**
1. Load the TrOCR processor and model (lazy-loaded, cached after first call).
2. For each line image, run the processor to create pixel values.
3. Generate text tokens with `model.generate()`, capturing output scores.
4. Compute per-token softmax probabilities and average them into a **line-level confidence score**.
5. Aggregate all line results into a `DocumentOCRResult`.

**Why TrOCR over alternatives:**
- **vs Tesseract:** TrOCR is transformer-based and trained specifically on handwriting; Tesseract struggles with cursive and irregular handwriting.
- **vs Google Vision API:** TrOCR runs locally with no API costs or network dependency.
- **vs EasyOCR:** TrOCR achieves higher word-error-rate on IAM benchmarks.

---

### Stage 3 — Text Post-processing (`pipeline/text_postprocessor.py`)

**Goal:** Clean up OCR artefacts and normalise the raw text.

**Steps taken:**
1. **Unicode Normalisation** — NFKC normalisation to canonical forms; smart quotes → ASCII quotes.
2. **Hyphenated Word Merging** — Rejoin words split across lines by hyphens (`hand-\nwriting` → `handwriting`).
3. **Common OCR Error Correction** — Pattern-based fixes for frequent confusions: `0` ↔ `o`, `1` ↔ `l`, `rn` ↔ `m`, doubled punctuation.
4. **Artefact Removal** — Strip isolated single-character noise tokens.
5. **Whitespace Collapse** — Normalise runs of spaces and blank lines.

**Why pattern-based correction:**
- Lightweight and deterministic — no extra model needed.
- Handles the most impactful error classes documented in OCR literature.
- Easily extensible with new patterns.

---

### Stage 4 — Chunking & Embedding (`pipeline/embeddings.py`, `pipeline/vector_store.py`)

**Goal:** Break text into searchable chunks and index them for semantic retrieval.

**Steps taken:**
1. **Chunking** — Split cleaned text into overlapping word-level chunks (default 512 words, 64-word overlap). Overlap ensures no information is lost at chunk boundaries.
2. **Embedding** — Encode each chunk using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors). This model balances quality and speed — it runs fast on CPU while providing strong semantic similarity.
3. **Indexing** — Store normalised embeddings in a FAISS `IndexFlatIP` (inner-product on L2-normalised vectors = cosine similarity). Flat index for exact search — appropriate for document-scale collections.
4. **Persistence** — The index and chunk metadata can be saved to disk and reloaded for future queries.

**Why FAISS over ChromaDB/Pinecone:**
- Zero external dependencies beyond `faiss-cpu`.
- Runs entirely offline.
- Flat index is simple, debuggable, and exact — no approximation errors for small-to-medium collections.

**Why all-MiniLM-L6-v2:**
- Best speed/quality trade-off on MTEB benchmarks for its size class.
- 384 dimensions keeps memory footprint small.
- Already a dependency in the parent project.

---

### Stage 5 — Retrieval-Augmented Structuring (`pipeline/rag_pipeline.py`)

**Goal:** Use RAG to retrieve the most relevant chunks and structure them into a coherent document.

**Steps taken:**
1. Formulate a structuring query based on the document title/filename.
2. Retrieve the top-k most relevant chunks from the FAISS index.
3. Concatenate retrieved chunks into a coherent text block.
4. Pass to the Document Structurer for formatting.

**RAG value-add:**
- When multiple documents are indexed, retrieval ensures only the relevant content is structured.
- Enables cross-document querying (e.g., "What were the meeting action items?").

---

### Stage 6 — Document Structuring (`pipeline/document_structurer.py`)

**Goal:** Detect document structure and output in multiple formats.

**Steps taken:**
1. **Heading Detection** — Regex patterns recognise Markdown-style headers (`#`), ALL-CAPS lines, numbered headings, and chapter/section markers.
2. **List Detection** — Identifies bulleted (`-`, `•`, `*`) and numbered (`1.`, `a)`) lists.
3. **Paragraph Grouping** — Consecutive non-heading, non-list lines are grouped into content sections.
4. **Multi-format Output:**
   - **JSON** — Full structured representation with title, sections, metadata.
   - **Markdown** — Clean, readable document with proper heading levels.
   - **Plain Text** — ASCII-formatted version with underlines for headings.

---

## Directory Structure

```
experiment_0/
├── README.md                  # This file
├── config.py                  # Pipeline configuration
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # CLI entry point
├── create_sample_images.py    # Generate test images
├── pipeline/
│   ├── __init__.py
│   ├── image_preprocessor.py  # Stage 1: Image cleaning
│   ├── handwriting_ocr.py     # Stage 2: TrOCR recognition
│   ├── text_postprocessor.py  # Stage 3: Text cleanup
│   ├── embeddings.py          # Stage 4a: Sentence embeddings
│   ├── vector_store.py        # Stage 4b: FAISS index
│   ├── document_structurer.py # Stage 6: Structure detection
│   └── rag_pipeline.py        # Orchestrator (all stages)
├── sample_images/             # Test images (generated)
└── output/                    # Structured output files
```

---

## Installation

```bash
# Navigate to experiment directory
cd experiment_0

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
- GPU optional but recommended for faster OCR (MPS on Apple Silicon is auto-detected)

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

### 5. Full Workflow (Generate + Process)

```bash
python run_pipeline.py --generate-samples --image-dir sample_images/ -v
```

The `-v` flag enables verbose logging for all pipeline stages.

### Output

Structured documents are saved to `output/` in three formats:
- `document.json` — Machine-readable structured format
- `document.md` — Human-readable Markdown
- `document.txt` — Plain-text version

---

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TROCR_MODEL` | `microsoft/trocr-large-handwritten` | HuggingFace model for OCR |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | 512 | Words per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 5 | Number of chunks to retrieve |
| `IMAGE_PREPROCESSING.denoise_strength` | 10 | NLMeans denoising strength |
| `IMAGE_PREPROCESSING.deskew` | True | Auto-straighten pages |

---

## Design Decisions & Methodology

### Why RAG for Document Digitisation?

Traditional OCR pipelines stop at text extraction. By adding a retrieval layer:
- Multiple documents become **searchable** via natural language.
- Retrieved context enables **smarter structuring** — the system focuses on relevant content.
- The architecture is **extensible** — swapping in an LLM for generation would enable summarisation, translation, or Q&A.

### Why Per-line OCR Instead of Full-page?

TrOCR was trained and evaluated on **single text lines** from the IAM dataset. Feeding it full pages degrades accuracy significantly. Our preprocessing pipeline segments pages into lines first, then runs OCR per-line — matching the model's training distribution.

### Why FAISS Instead of a Full Vector Database?

For this experiment's scale (single-user, local processing, hundreds-to-thousands of chunks), FAISS `IndexFlatIP` provides:
- **Exact** nearest-neighbour search (no approximation).
- **Zero operational overhead** (no server, no network).
- **Simple persistence** (one `.faiss` file + one `.json` file).

### Why Sentence-Transformers for Embeddings?

`all-MiniLM-L6-v2` delivers strong semantic similarity at 384 dimensions — small enough to keep in memory but powerful enough to distinguish topically different chunks. It's also already a dependency of the parent benchmarking project.

---

## Libraries & Tools Used

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.2.0 | Deep learning runtime |
| `transformers` | ≥4.40.0 | TrOCR model loading & inference |
| `sentence-transformers` | ≥2.7.0 | Text embedding |
| `opencv-python` | ≥4.9.0 | Image preprocessing (denoise, binarise, deskew) |
| `Pillow` | ≥10.2.0 | Image I/O and manipulation |
| `faiss-cpu` | ≥1.8.0 | Vector similarity search |
| `numpy` | ≥1.26.0 | Numerical operations |

---

## Limitations & Future Work

1. **English-only** — TrOCR-Large-Handwritten is trained on English. Multilingual support would require `microsoft/trocr-large-handwritten` fine-tuned on other languages or a multilingual model.
2. **No LLM generation step** — The current pipeline structures text via rule-based heuristics. Adding an LLM (e.g., GPT-4, Llama) for the "Generate" step would enable summarisation, entity extraction, and Q&A.
3. **Sequential processing** — Line-by-line OCR is not batched. Batching would improve throughput on GPU.
4. **Layout analysis** — Complex layouts (tables, multi-column, diagrams) are not handled. Integrating a layout analysis model (e.g., LayoutLMv3) is a natural extension.
5. **Confidence-based rejection** — Low-confidence lines could be flagged for human review rather than silently included.

---

*Experiment 0 — Built as part of the Coding Agent Search Benchmark project.*
