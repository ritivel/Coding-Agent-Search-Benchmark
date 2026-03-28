"""
Main RAG pipeline orchestrator for handwritten document digitisation.

Ties together every pipeline component:
  Image → Preprocess → OCR → Post-process → Embed/Index → Retrieve → Structure

The pipeline can operate in two modes:
  1. **Ingest** — process document images and build the vector store.
  2. **Query** — retrieve relevant chunks and produce a structured document.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .document_structurer import DocumentStructurer, StructuredDocument
from .embeddings import EmbeddingEngine
from .handwriting_ocr import DocumentOCRResult, HandwritingOCR
from .image_preprocessor import ImagePreprocessor
from .text_postprocessor import TextPostProcessor
from .vector_store import Chunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from a single document processing run."""

    source_path: str
    ocr_result: DocumentOCRResult | None = None
    cleaned_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    structured_document: StructuredDocument | None = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


class HandwritingRAGPipeline:
    """
    End-to-end RAG pipeline for handwritten document images.

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
        device: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        preprocessing_config: dict | None = None,
        persist_dir: Path | None = None,
    ):
        pp_cfg = preprocessing_config or {}
        self.preprocessor = ImagePreprocessor(**pp_cfg)
        self.ocr = HandwritingOCR(model_name=trocr_model, device=device)
        self.postprocessor = TextPostProcessor()
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.vector_store = VectorStore(
            embedding_engine=self.embedding_engine,
            persist_dir=persist_dir,
        )
        self.structurer = DocumentStructurer()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def process_image(
        self,
        image_path: str | Path,
        output_dir: Path | None = None,
        title: str = "",
        output_formats: list[str] | None = None,
    ) -> PipelineResult:
        """
        Full pipeline: image → structured document.

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

        # --- Step 1: Image Preprocessing ---
        t0 = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(image_path)
        result.timings["preprocessing"] = time.perf_counter() - t0
        logger.info(
            "Preprocessing done: %d line images (%.2fs)",
            len(preprocessed["line_images"]),
            result.timings["preprocessing"],
        )

        # --- Step 2: Handwriting OCR ---
        t0 = time.perf_counter()
        ocr_result = self.ocr.recognise_document(
            preprocessed["line_images"],
            source_path=str(image_path),
        )
        result.ocr_result = ocr_result
        result.timings["ocr"] = time.perf_counter() - t0
        logger.info("OCR done: %d lines (%.2fs)", len(ocr_result.lines), result.timings["ocr"])

        # --- Step 3: Text Post-processing ---
        t0 = time.perf_counter()
        cleaned = self.postprocessor.process(ocr_result.raw_text)
        result.cleaned_text = cleaned
        result.timings["postprocessing"] = time.perf_counter() - t0

        # --- Step 4: Chunk & Index ---
        t0 = time.perf_counter()
        chunks = self.vector_store.add_text(
            cleaned,
            source=str(image_path),
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        result.chunks = chunks
        result.timings["indexing"] = time.perf_counter() - t0

        # --- Step 5: Retrieval-augmented structuring ---
        t0 = time.perf_counter()
        query = f"Structure the content of the handwritten document: {title or image_path.stem}"
        retrieved = self.vector_store.search(query, top_k=self.top_k)
        retrieved_text = "\n\n".join(chunk.text for chunk, _score in retrieved)

        structuring_input = retrieved_text if retrieved_text.strip() else cleaned

        metadata = {
            "source": str(image_path),
            "ocr_lines": len(ocr_result.lines),
            "total_chars": len(cleaned),
            "chunks": len(chunks),
        }
        structured = self.structurer.structure(
            structuring_input, title=title or image_path.stem, metadata=metadata
        )
        result.structured_document = structured
        result.timings["structuring"] = time.perf_counter() - t0

        # --- Step 6: Save outputs ---
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

    def query(self, question: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Query the indexed documents (after ingestion)."""
        k = top_k or self.top_k
        return self.vector_store.search(question, top_k=k)

    def save_index(self, directory: Path | None = None) -> None:
        self.vector_store.save(directory)

    def load_index(self, directory: Path | None = None) -> None:
        self.vector_store.load(directory)
