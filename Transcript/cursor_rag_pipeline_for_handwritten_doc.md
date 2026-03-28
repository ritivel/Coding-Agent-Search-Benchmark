# RAG pipeline for handwritten document processing
_Exported on 28/03/2026 at 21:34:20 GMT+5:30 from Cursor (2.6.20)_

---

**User**

This is experiment 3.

Build a RAG pipeline that takes handwritten document images as input, extracts text, and produces a structured digital document as output. Use The skill.md file in experiment_3 folder to find the best methodology to create a RAG. use experiment_3  folder for this experiment.And also create a README file explaining all the steps that you had taken.To develop this particular RAG pipeline

---

**Cursor**

Let me start by reading the SKILL.md file to understand the methodology, then I'll build the RAG pipeline.

I've read the SKILL.md. It prescribes a **research-first workflow** before building. Let me follow it step by step.

**Phase 1: Understanding the Request**
- **Core problem:** Build a RAG pipeline that takes handwritten document images, extracts text via OCR, and produces structured digital documents.
- **Key technical requirements:** Python, OCR (handwriting-capable), vector store, LLM integration, document structuring.
- **Quality bar:** Functional prototype with clear pipeline stages.

Now let me execute Phases 2-4 by researching the web for the best existing implementations, libraries, and documentation.

Excellent research results. Let me dig deeper into the specific libraries and tools.

Now I have comprehensive research. Let me synthesize the findings and build the pipeline.

Let me start building all the modules. I'll create the project structure and all files.

Now let me create the `.env.example` file and the comprehensive README.

Now let me write the comprehensive README.

Now let me create a `RESEARCH.md` file as recommended by the SKILL.md, to preserve the research findings for future developers.

Now let me verify all the files are properly created and check for any lint issues.

All tasks are complete. Here's a summary of what was built in the `experiment_3/` folder:

---

## What Was Built

A complete **Handwritten Document RAG Pipeline** with 8 modular Python files, following the research-first methodology from `SKILL.md`.

### Pipeline Flow

```
Handwritten Image → Preprocessing → Dual OCR → Text Cleaning → Chunking → Embedding → FAISS Store → LLM Structuring → JSON/Markdown/Text Output
```

### Files Created

| File | Purpose |
|---|---|
| `config.py` | Centralized configuration via environment variables |
| `preprocessing.py` | 5-stage image preprocessing (grayscale → denoise → deskew → threshold → rescale) |
| `ocr_engine.py` | Dual OCR: Tesseract (fast/bulk) + TrOCR transformer (handwriting-optimized), with per-line merge |
| `text_processing.py` | Unicode normalization, OCR artefact fixing, recursive text chunking |
| `vector_store.py` | FAISS vector store with sentence-transformers embeddings (all-MiniLM-L6-v2) |
| `rag_pipeline.py` | RAG orchestrator with two modes: document structuring + Q&A |
| `structured_output.py` | Multi-format output (JSON with full metadata, Markdown, plain text) |
| `main.py` | CLI with `ingest` (process images) and `query` (ask questions) commands |
| `requirements.txt` | All Python dependencies |
| `.env.example` | Configuration template |
| `RESEARCH.md` | Preserved research findings (GitHub repos, library comparisons, docs) |
| `README.md` | Comprehensive documentation covering all 9 development steps |

### Key Technology Choices (Research-Informed)

- **Tesseract + TrOCR** dual OCR — best of classical speed and transformer handwriting accuracy
- **FAISS** — no server needed, fastest local vector search
- **sentence-transformers (all-MiniLM-L6-v2)** — free, fast, high-quality 384-dim embeddings
- **LangChain** — mature RAG orchestration framework
- **GPT-4o-mini** primary / **flan-t5-base** local fallback for LLM generation

### Usage

```bash
# Process a handwritten image
python main.py ingest --image sample_images/letter.png

# OCR-only mode (no LLM needed)
python main.py ingest --image photo.jpg --no-llm

# Ask questions about processed documents
python main.py query "What is the date in the letter?"
```

