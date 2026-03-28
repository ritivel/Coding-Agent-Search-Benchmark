"""
Main RAG pipeline orchestrator for handwritten document digitisation.

Ties together every pipeline component:
  Image → Preprocess → OCR → Post-process → Chunk → Embed/Index → Hybrid Retrieve → Re-rank → Structure

Key differences from Experiment 0:
  - ChromaDB vector store (persistent, auto-managed)
  - Hybrid BM25+vector retrieval with Reciprocal Rank Fusion
  - Cross-encoder re-ranking for precision
  - Recursive character splitting with metadata enrichment
  - Morphological-enhanced image preprocessing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk, RecursiveChunker
from .document_structurer import DocumentStructurer, StructuredDocument
from .embeddings import EmbeddingEngine
from .handwriting_ocr import DocumentOCRResult, HandwritingOCR
from .image_preprocessor import ImagePreprocessor
from .retriever import HybridRetriever, RetrievalResult
from .text_postprocessor import TextPostProcessor
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from a single document processing run."""

    source_path: str
    ocr_result: DocumentOCRResult | None = None
    cleaned_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    structured_document: StructuredDocument | None = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


class HandwritingRAGPipeline:
    """End-to-end advanced RAG pipeline for handwritten document images.

    Usage
    -----
    >>> pipeline = HandwritingRAGPipeline()
    >>> result = pipeline.process_image("scan.jpg", output_dir=Path("output"))
    >>> print(result.structured_document.to_markdown())
    """

    def __init__(
        self,
        trocr_model: str = "microsoft/trocr-large-handwritten",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunk_separators: list[str] | None = None,
        top_k: int = 5,
        rerank_candidates: int = 20,
        hybrid_alpha: float = 0.7,
        rrf_k: int = 60,
        preprocessing_config: dict | None = None,
        chroma_persist_dir: Path | None = None,
        collection_name: str = "handwritten_docs",
    ):
        pp_cfg = preprocessing_config or {}
        self.preprocessor = ImagePreprocessor(**pp_cfg)
        self.ocr = HandwritingOCR(model_name=trocr_model, device=device)
        self.postprocessor = TextPostProcessor()
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=chunk_separators,
        )
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.vector_store = ChromaVectorStore(
            embedding_engine=self.embedding_engine,
            persist_dir=chroma_persist_dir,
            collection_name=collection_name,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            reranker_model=reranker_model,
            alpha=hybrid_alpha,
            rrf_k=rrf_k,
        )
        self.structurer = DocumentStructurer()
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def process_image(
        self,
        image_path: str | Path,
        output_dir: Path | None = None,
        title: str = "",
        output_formats: list[str] | None = None,
    ) -> PipelineResult:
        """Full pipeline: image → structured document.

        Parameters
        ----------
        image_path : path to a handwritten document image.
        output_dir : where to write output files (optional).
        title      : title for the structured document.
        output_formats : list of output formats ("json", "markdown", "txt").

        Returns
        -------
        PipelineResult with all intermediate and final artefacts.
        """
        image_path = Path(image_path)
        result = PipelineResult(source_path=str(image_path))
        output_formats = output_formats or ["json", "markdown", "txt"]

        # --- Stage 1: Image Preprocessing (with morphological cleanup) ---
        t0 = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(image_path)
        result.timings["preprocessing"] = time.perf_counter() - t0
        logger.info(
            "Preprocessing done: %d line images (%.2fs)",
            len(preprocessed["line_images"]),
            result.timings["preprocessing"],
        )

        # --- Stage 2: Handwriting OCR (TrOCR with batch support) ---
        t0 = time.perf_counter()
        ocr_result = self.ocr.recognise_document(
            preprocessed["line_images"],
            source_path=str(image_path),
        )
        result.ocr_result = ocr_result
        result.timings["ocr"] = time.perf_counter() - t0
        logger.info("OCR done: %d lines (%.2fs)", len(ocr_result.lines), result.timings["ocr"])

        # --- Stage 3: Text Post-processing ---
        t0 = time.perf_counter()
        cleaned = self.postprocessor.process(ocr_result.raw_text)
        result.cleaned_text = cleaned
        result.timings["postprocessing"] = time.perf_counter() - t0

        # --- Stage 4: Recursive Chunking with metadata ---
        t0 = time.perf_counter()
        chunks = self.chunker.split_text(
            cleaned,
            source=str(image_path),
            ocr_confidence=ocr_result.avg_confidence,
        )
        result.chunks = chunks
        result.timings["chunking"] = time.perf_counter() - t0

        # --- Stage 5: Embed & Index in ChromaDB ---
        t0 = time.perf_counter()
        self.vector_store.add_chunks(chunks)
        result.timings["indexing"] = time.perf_counter() - t0

        # --- Stage 6: Hybrid Retrieval + Re-ranking ---
        t0 = time.perf_counter()
        query = f"Structure the content of the handwritten document: {title or image_path.stem}"
        retrieval_results = self.retriever.search(
            query,
            top_k=self.top_k,
            rerank_candidates=self.rerank_candidates,
        )
        result.retrieval_results = retrieval_results
        result.timings["retrieval"] = time.perf_counter() - t0

        retrieved_text = "\n\n".join(r.chunk.text for r in retrieval_results)
        structuring_input = retrieved_text if retrieved_text.strip() else cleaned

        # --- Stage 7: Document Structuring ---
        t0 = time.perf_counter()
        metadata = {
            "source": str(image_path),
            "ocr_lines": len(ocr_result.lines),
            "total_chars": len(cleaned),
            "chunks": len(chunks),
            "avg_ocr_confidence": f"{ocr_result.avg_confidence:.3f}",
            "retrieval_method": retrieval_results[0].method if retrieval_results else "none",
        }
        structured = self.structurer.structure(
            structuring_input, title=title or image_path.stem, metadata=metadata
        )
        result.structured_document = structured
        result.timings["structuring"] = time.perf_counter() - t0

        # --- Stage 8: Save outputs ---
        if output_dir:
            t0 = time.perf_counter()
            basename = image_path.stem
            result.output_paths = self.structurer.save(
                structured, output_dir, basename=basename, formats=output_formats
            )
            result.timings["saving"] = time.perf_counter() - t0

        total = sum(result.timings.values())
        result.timings["total"] = total
        logger.info("Pipeline complete in %.2fs", total)
        return result

    def process_batch(
        self,
        image_paths: list[str | Path],
        output_dir: Path | None = None,
        output_formats: list[str] | None = None,
    ) -> list[PipelineResult]:
        """Process multiple document images sequentially."""
        results: list[PipelineResult] = []
        for path in image_paths:
            result = self.process_image(
                path, output_dir=output_dir, output_formats=output_formats
            )
            results.append(result)
        return results

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_reranking: bool = True,
    ) -> list[RetrievalResult]:
        """Query the indexed documents using hybrid retrieval + re-ranking."""
        k = top_k or self.top_k
        return self.retriever.search(
            question,
            top_k=k,
            rerank_candidates=self.rerank_candidates,
            use_reranking=use_reranking,
        )

    def query_simple(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Query using vector-only search (no hybrid/reranking)."""
        k = top_k or self.top_k
        return self.retriever.vector_only_search(question, top_k=k)
