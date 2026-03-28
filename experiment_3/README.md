# Experiment 3 — Handwritten Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that takes **handwritten document images** as input, extracts text using dual OCR engines, and produces a **structured digital document** as output.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Research Methodology](#research-methodology)
4. [Development Steps](#development-steps)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Pipeline Deep Dive](#pipeline-deep-dive)
10. [Technology Choices & Rationale](#technology-choices--rationale)
11. [Output Formats](#output-formats)
12. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This pipeline solves the problem of converting handwritten documents (letters, notes, forms, manuscripts) from images into clean, structured digital text. It combines classical computer vision preprocessing, dual OCR engines (Tesseract + TrOCR), semantic chunking, vector storage, and LLM-based text structuring.

**Input:** Handwritten document image (PNG, JPG, TIFF, BMP, WebP)  
**Output:** Structured digital document in JSON, Markdown, and plain text formats

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Handwritten │     │  Image           │     │  Dual OCR       │
│  Document    │────▶│  Preprocessing   │────▶│  Engine         │
│  Image       │     │  (OpenCV)        │     │  (Tesseract +   │
└─────────────┘     └──────────────────┘     │   TrOCR)        │
                                              └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Structured  │     │  RAG Pipeline    │     │  Text Processing│
│  Digital     │◀────│  (Retrieval +    │◀────│  (Clean, Chunk, │
│  Document    │     │   Generation)    │     │   Embed, Store) │
└─────────────┘     └──────────────────┘     └─────────────────┘
      │
      ▼
 JSON / Markdown / Plain Text
```

---

## Research Methodology

Following the **web-research-before-building** skill (see `SKILL.md`), we conducted systematic research before writing code.

### Phase 1 — Problem Definition

- **Core problem:** Convert handwritten document images into structured digital text via a RAG pipeline.
- **Key requirements:** Python, handwriting-capable OCR, vector store, LLM for structuring, multi-format output.
- **Quality bar:** Functional prototype with modular, extensible architecture.

### Phase 2 — GitHub Research (Existing Implementations)

We searched GitHub for existing RAG + OCR pipelines and found these key references:

| Repository | Stars | Key Takeaway |
|---|---|---|
| [nsourlos/OCR_and_RAG](https://github.com/nsourlos/OCR_and_RAG) | — | Benchmarked 15+ OCR tools; recommends GPT-4o Vision for equations/complex layouts |
| [revoker3661/AI-Document-Intelligence-Pipeline](https://github.com/revoker3661/AI-Document-Intelligence-Pipeline) | — | Production pipeline with Detectron2 layout analysis + PaddleOCR |
| [Clay-111/Multimodal-RAG-PDFs](https://github.com/Clay-111/Multimodal-RAG-PDFs) | — | Multimodal RAG with Pinecone + Streamlit; supports OCR for Bangla/English |
| [novatechflow/docai](https://github.com/2pk03/docai) | — | Local-first OCR → Markdown → RAG with FAISS and Tkinter UI |
| [ShashvatRajora/RAG_implemented_ChatBot](https://github.com/ShashvatRajora/RAG_implemented_ChatBot) | — | Dual FAISS stores for text + tables; CNN-based table detection |

### Phase 3 — Library Research

| Library | Purpose | Why Chosen |
|---|---|---|
| **Tesseract (pytesseract)** | Classical OCR | Proven, supports 100+ languages, good on printed text, fast |
| **TrOCR (microsoft/trocr-base-handwritten)** | Transformer OCR | State-of-the-art for handwriting; encoder-decoder ViT+RoBERTa architecture |
| **OpenCV** | Image preprocessing | Industry standard for image manipulation, deskewing, denoising |
| **sentence-transformers (all-MiniLM-L6-v2)** | Text embeddings | Fast, lightweight, excellent quality for semantic similarity |
| **FAISS** | Vector store | Facebook's library; blazing fast similarity search, no external server needed |
| **LangChain** | RAG orchestration | Industry-standard framework for text splitting, vector stores, LLM chains |
| **Pydantic** | Data models | Clean structured output with validation |

### Phase 4 — Documentation & Best Practices

- **Tesseract best practices:** 300+ DPI, adaptive thresholding, deskewing, denoising (from [tessdoc](https://tesseract-ocr.github.io/tessdoc/ImproveQuality))
- **TrOCR usage:** HuggingFace Transformers library, line-by-line processing for best accuracy (from [HF docs](https://huggingface.co/docs/transformers/en/model_doc/trocr))
- **FAISS + LangChain:** From [LangChain FAISS integration docs](https://docs.langchain.com/oss/python/integrations/vectorstores/faiss)

### Phase 5 — Architecture Decision

We chose a **dual-OCR merge strategy** — Tesseract for speed/coverage and TrOCR for handwriting accuracy — because no single OCR engine excels at all handwriting styles. The merge step selects the higher-confidence result per line.

---

## Development Steps

Here is a chronological account of every step taken to build this pipeline:

### Step 1: Project Setup & Dependency Selection

Created `requirements.txt` with pinned minimum versions for all dependencies:
- Core OCR: `pytesseract`, `Pillow`, `opencv-python`
- Transformer OCR: `transformers`, `torch`
- Text processing: `langchain`, `langchain-text-splitters`
- Embeddings & vector store: `sentence-transformers`, `faiss-cpu`
- Structured output: `pydantic`
- LLM: `langchain-openai`, `openai` (optional)
- Utilities: `numpy`, `tqdm`, `python-dotenv`

### Step 2: Configuration Module (`config.py`)

Built a centralized configuration system using environment variables (loaded via `python-dotenv`) with sensible defaults. This covers:
- OCR engine selection (Tesseract PSM mode, TrOCR model name)
- Image preprocessing parameters (DPI target, denoise strength, threshold block size)
- Text chunking parameters (chunk size: 500 chars, overlap: 100 chars)
- Embedding model selection (all-MiniLM-L6-v2 by default)
- LLM provider toggle (OpenAI vs. local HuggingFace model)
- RAG retrieval parameters (top-k: 5, similarity threshold)

### Step 3: Image Preprocessing Module (`preprocessing.py`)

Built a 5-stage image preprocessing pipeline optimized for handwritten text:

1. **Load & grayscale conversion** — `cv2.imread` + `cv2.cvtColor`
2. **Non-local means denoising** — `cv2.fastNlMeansDenoising` preserves handwriting strokes better than Gaussian blur
3. **Deskewing** — Uses `cv2.minAreaRect` on detected contours to straighten tilted documents; only rotates if angle > 0.5°
4. **Adaptive Gaussian thresholding** — `cv2.adaptiveThreshold` for clean black-on-white binarization that handles uneven illumination
5. **Rescaling** — Scales image height to 3000px (preserving aspect ratio) to meet Tesseract's 300+ DPI recommendation

Returns both an OpenCV numpy array (for Tesseract) and a PIL Image (for TrOCR).

### Step 4: Dual OCR Engine (`ocr_engine.py`)

Implemented two OCR engines and a merge strategy:

**Tesseract OCR:**
- Uses `pytesseract.image_to_data()` for word-level output with confidence scores
- Aggregates words into lines, computing per-line average confidence
- Configurable PSM mode (default: 6 — uniform block of text)

**TrOCR (Transformer-based):**
- Uses Microsoft's `trocr-base-handwritten` model via HuggingFace Transformers
- Implements horizontal-projection line segmentation: projects pixel intensities, finds gaps between text lines, crops individual lines
- Processes each line crop independently through the ViT encoder → RoBERTa decoder
- Lazy-loads the model to avoid startup cost when not needed

**Merge Strategy:**
- Aligns lines from both engines by position
- For each line pair, selects the result with higher confidence
- Falls back to whichever engine produced the line if the other missed it

### Step 5: Text Processing Module (`text_processing.py`)

Built text cleaning and chunking:

**Cleaning pipeline:**
- Unicode NFC normalisation + invisible character removal
- Optional OCR artefact correction (common Tesseract confusions like `|` → `I`, `rn` → `m`)
- Whitespace collapse (multiple spaces → one, excessive newlines → double)

**Chunking:**
- Uses LangChain's `RecursiveCharacterTextSplitter` with hierarchical separators: `\n\n` → `\n` → `. ` → ` ` → `""`
- Default chunk size: 500 characters with 100-character overlap
- Tracks character offsets for each chunk back to the original text

### Step 6: Vector Store Module (`vector_store.py`)

Built FAISS-based vector storage:

**EmbeddingWrapper class:**
- Wraps `sentence-transformers` to satisfy LangChain's Embeddings interface (`embed_documents` / `embed_query`)
- Uses `all-MiniLM-L6-v2` (384-dimension, ~80MB) — fast and high-quality

**VectorStoreManager class:**
- `build_from_chunks()` — Creates a new FAISS index from text chunks with metadata (source file, chunk index, character offsets)
- `add_chunks()` — Incrementally adds to existing index
- `search()` — Similarity search returning (document, score) pairs
- `save()` / `load()` — Persist to / restore from disk for session continuity

### Step 7: RAG Pipeline Orchestrator (`rag_pipeline.py`)

Built the core RAG logic with two modes:

**Structuring mode (`structure_document`):**
- Takes raw OCR text and sends it to the LLM with a detailed prompt
- The prompt instructs the LLM to: correct OCR errors, preserve original meaning, organize with headings/paragraphs/lists, reproduce tables in Markdown format
- Outputs ONLY the structured document — no commentary

**Q&A mode (`query`):**
- Retrieves top-k relevant chunks from the vector store
- Assembles them as context in a QA prompt
- Returns the LLM answer plus source attribution (chunk indices, scores, text previews)

**LLM configuration:**
- Primary: OpenAI GPT-4o-mini (fast, cheap, high quality)
- Fallback: Local HuggingFace `google/flan-t5-base` (no API key needed, runs on CPU)

### Step 8: Structured Output Module (`structured_output.py`)

Built the final output generation:

**StructuredDocument dataclass:**
- Contains: title, source image path, OCR engine used, confidence score, raw text, structured text, extracted sections, metadata, timestamp

**Section parsing:**
- Scans LLM output for Markdown headings (`## Heading`) and splits into `DocumentSection` objects

**Export formats:**
- **JSON** — Full metadata, sections, raw text, timestamps (machine-readable)
- **Markdown** — Formatted document with metadata header, sections, and collapsible raw OCR text
- **Plain text** — Clean text with section headings (simple consumption)

### Step 9: CLI Entry Point (`main.py`)

Built an `argparse`-based CLI with two commands:

**`ingest` command:**
- Accepts `--image` (single file) or `--dir` (directory of images)
- Runs the full 5-stage pipeline per image: preprocess → OCR → clean/chunk → embed/store → structure
- Persists the vector store to disk for later queries
- Prints a summary table of all processed images

**`query` command:**
- Loads the persisted vector store
- Runs RAG Q&A and prints the answer with source attribution
- Configurable `--top-k` for retrieval depth

---

## Project Structure

```
experiment_3/
├── SKILL.md              # Research-first methodology (the skill used)
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── config.py             # Centralized configuration
├── preprocessing.py      # Image preprocessing (OpenCV)
├── ocr_engine.py         # Dual OCR: Tesseract + TrOCR
├── text_processing.py    # Text cleaning & chunking
├── vector_store.py       # FAISS vector store management
├── rag_pipeline.py       # RAG orchestration (retrieval + generation)
├── structured_output.py  # Multi-format document output
├── main.py               # CLI entry point
├── sample_images/        # Place handwritten images here
├── output/               # Generated structured documents (auto-created)
└── vector_store/         # Persisted FAISS index (auto-created)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Tesseract OCR installed on your system

**Install Tesseract:**

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# Windows — download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Python Setup

```bash
cd experiment_3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings (e.g., OPENAI_API_KEY)
```

---

## Usage

### 1. Process a single handwritten image

```bash
python main.py ingest --image sample_images/letter.png
```

### 2. Process all images in a directory

```bash
python main.py ingest --dir sample_images/
```

### 3. OCR-only mode (no LLM required)

```bash
python main.py ingest --image sample_images/letter.png --no-llm
```

### 4. Ask questions about ingested documents

```bash
python main.py query "What is the date mentioned in the document?"
python main.py query "Who signed the letter?" --top-k 3
```

---

## Configuration

All settings are configurable via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `local` (HuggingFace) |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `LOCAL_LLM_MODEL` | `google/flan-t5-base` | Local model for fallback |
| `TESSERACT_CMD` | `tesseract` | Path to Tesseract binary |
| `TESSERACT_LANG` | `eng` | OCR language |
| `TESSERACT_PSM` | `6` | Page segmentation mode |
| `TROCR_MODEL` | `microsoft/trocr-base-handwritten` | TrOCR model |
| `USE_TROCR` | `true` | Enable/disable TrOCR |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |

---

## Pipeline Deep Dive

### Stage 1: Image Preprocessing

```
Raw Image → Grayscale → Denoise → Deskew → Threshold → Rescale
```

- **Grayscale:** Reduces 3-channel colour to 1-channel intensity
- **Denoise:** Non-local means algorithm (preserves edges/strokes better than Gaussian blur)
- **Deskew:** Detects skew angle via `minAreaRect` on contour points; rotates to correct
- **Threshold:** Adaptive Gaussian produces clean black-on-white regardless of lighting
- **Rescale:** Upscales small images to 3000px height for Tesseract's 300+ DPI sweet spot

### Stage 2: Dual OCR

```
Preprocessed Image ─┬─▶ Tesseract (word-by-word + confidence)
                    └─▶ TrOCR (line-by-line + transformer decoding)
                           │
                           ▼
                    Merge (higher confidence wins per line)
```

### Stage 3: Text Processing

```
Raw OCR Text → Unicode normalisation → Artefact fixing → Whitespace collapse → Chunking
```

### Stage 4: Embedding & Storage

```
Chunks → sentence-transformers → 384-dim vectors → FAISS IndexFlatL2
```

### Stage 5: RAG Generation

```
Query → Embed → FAISS search → Top-K chunks → LLM prompt → Structured output
```

---

## Technology Choices & Rationale

| Decision | Choice | Alternative Considered | Reason |
|---|---|---|---|
| Primary OCR | Tesseract | PaddleOCR, EasyOCR | Widest language support, most mature, fastest for batch |
| Handwriting OCR | TrOCR | Thulium, Kraken | Transformer architecture gives best handwriting accuracy; HuggingFace integration |
| Vector Store | FAISS | Pinecone, Qdrant, ChromaDB | No server required, fastest for local use, Facebook-backed |
| Embeddings | all-MiniLM-L6-v2 | OpenAI ada-002, BGE | Free, fast, 384-dim is efficient, excellent quality |
| Text Splitter | RecursiveCharacterTextSplitter | TokenTextSplitter | Respects paragraph/sentence boundaries for better semantic chunks |
| LLM | GPT-4o-mini (primary) | GPT-4o, Claude, Llama | Best cost/quality ratio; local fallback via flan-t5 |
| Framework | LangChain | LlamaIndex, raw code | Mature vector store integrations, text splitters, LLM chains |

---

## Output Formats

### JSON Output

```json
{
  "title": "Handwritten Letter",
  "source_image": "letter.png",
  "ocr_engine": "merged",
  "ocr_confidence": 78.5,
  "raw_text": "Dear Sir, I am writing to ...",
  "structured_text": "# Letter\n\nDear Sir,\n\nI am writing to ...",
  "sections": [
    { "heading": "Letter", "content": "Dear Sir, ...", "order": 0 }
  ],
  "metadata": {
    "char_count_raw": 450,
    "char_count_structured": 520,
    "section_count": 1,
    "num_chunks": 3
  },
  "created_at": "2026-03-28T12:00:00+00:00"
}
```

### Markdown Output

A formatted document with metadata header, structured content with headings, and a collapsible raw OCR text section for verification.

### Plain Text Output

Clean text with section headings, suitable for simple downstream consumption.

---

## Limitations & Future Work

### Current Limitations

- **Line segmentation:** The horizontal-projection method for TrOCR line segmentation works well for clearly separated lines but may struggle with overlapping handwriting
- **Single language:** Currently optimised for English; multi-language support requires changing `TESSERACT_LANG` and using language-specific TrOCR models
- **No table detection:** Complex tables in handwritten documents are treated as regular text
- **LLM dependency:** Full structuring requires an LLM; the `--no-llm` mode outputs cleaned but unstructured text

### Future Improvements

- **Layout analysis:** Add Detectron2 or LayoutLMv3 for detecting text regions, tables, and figures
- **Table extraction:** Integrate Microsoft Table Transformer for structured table output
- **Multi-language TrOCR:** Support for non-English handwriting models
- **Confidence-based routing:** Automatically decide Tesseract vs TrOCR per region based on handwriting detection
- **Streaming pipeline:** Process very large document batches with async/streaming architecture
- **Web UI:** Add a Streamlit or Gradio interface for interactive document processing
