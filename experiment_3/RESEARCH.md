# Research Summary — Handwritten Document RAG Pipeline

> This file preserves the web research conducted before building the pipeline,
> following the `web-research-before-building` skill.

## Recommended Approach

Use a **dual-OCR strategy** (Tesseract for speed + TrOCR for handwriting accuracy) with
FAISS vector storage and LLM-based structuring. This gives the best trade-off between
accuracy, speed, and deployment simplicity (no external services required beyond an
optional LLM API).

## Best Existing Implementations Found

1. **[nsourlos/OCR_and_RAG](https://github.com/nsourlos/OCR_and_RAG)** — June 2025
   - Why it's relevant: Benchmarked 15+ OCR tools for PDF RAG pipelines
   - Key takeaway: GPT-4o Vision best for complex layouts; PaddleOCR and Surya-OCR
     are strong open-source contenders

2. **[revoker3661/AI-Document-Intelligence-Pipeline](https://github.com/revoker3661/AI-Document-Intelligence-Pipeline)** — Dec 2025
   - Why it's relevant: Production-grade medical PDF → structured JSONL pipeline
   - Key takeaway: Detectron2 for layout analysis + PaddleOCR for GPU-accelerated OCR

3. **[Clay-111/Multimodal-RAG-PDFs](https://github.com/Clay-111/Multimodal-RAG-PDFs)** — Jan 2026
   - Why it's relevant: Multimodal RAG with OCR + vision-language models
   - Key takeaway: Pinecone for cloud vector store; Streamlit for UI

4. **[novatechflow/docai](https://github.com/2pk03/docai)** — Dec 2025
   - Why it's relevant: Local-first OCR → Markdown → RAG with FAISS
   - Key takeaway: Pluggable OCR/embeddings/LLM components pattern

5. **[ShashvatRajora/RAG_implemented_ChatBot](https://github.com/ShashvatRajora/RAG_implemented_ChatBot)** — Dec 2025
   - Why it's relevant: Dual FAISS stores (text + tables), CNN table detection
   - Key takeaway: Separate vector stores for different content types

## Key Libraries Selected

| Library | Why Over Alternatives |
|---|---|
| **pytesseract** | Most mature OCR wrapper; 100+ languages; fast batch processing |
| **TrOCR (trocr-base-handwritten)** | Best handwriting accuracy via ViT+RoBERTa transformer arch |
| **FAISS** | No server needed; fastest local similarity search; Facebook-backed |
| **all-MiniLM-L6-v2** | Free, 384-dim, fast, excellent semantic quality |
| **LangChain** | Best ecosystem for text splitting + vector stores + LLM chaining |
| **OpenCV** | Standard for image preprocessing; adaptive threshold + deskew |

## Documentation References

- [Tesseract: Improving Quality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality) — preprocessing best practices
- [TrOCR HuggingFace Docs](https://huggingface.co/docs/transformers/en/model_doc/trocr) — model usage
- [LangChain FAISS Integration](https://docs.langchain.com/oss/python/integrations/vectorstores/faiss) — vector store setup
- [sentence-transformers Docs](https://www.sbert.net/) — embedding models

## Risks / Open Questions

- TrOCR processes line-by-line: line segmentation quality directly impacts accuracy
- Local LLM fallback (flan-t5-base) produces lower-quality structuring than GPT-4o-mini
- Very cursive or non-English handwriting may need specialised TrOCR models
