"""RAG Pipeline Orchestrator — ties all stages together."""

import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    source_path: str
    ocr_text: str
    cleaned_text: str
    ocr_confidence: float
    chunk_count: int
    structured_document: object = None


@dataclass
class QueryResult:
    query: str
    retrieved_chunks: list = field(default_factory=list)
    reranked_chunks: list = field(default_factory=list)
    answer_context: str = ""


class HandwrittenDocumentRAGPipeline:
    """End-to-end pipeline: Image -> OCR -> Clean -> Chunk -> Embed -> Index -> Retrieve -> Structure."""

    def __init__(self, config=None):
        if config is None:
            from experiment_2 import config as cfg
            config = cfg

        self.config = config

        from .image_preprocessor import ImagePreprocessor
        from .handwriting_ocr import SuryaOCREngine
        from .text_postprocessor import TextPostProcessor
        from .embeddings import SemanticChunker
        from .vector_store import ChromaVectorStore
        from .reranker import CrossEncoderReranker
        from .document_structurer import DocumentStructurer

        self.preprocessor = ImagePreprocessor(**config.IMAGE_PREPROCESSING)
        self.ocr_engine = SuryaOCREngine(languages=config.SURYA_LANGUAGES)
        self.text_processor = TextPostProcessor()
        self.chunker = SemanticChunker(
            model_name=config.EMBEDDING_MODEL,
            max_chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            similarity_threshold=config.SEMANTIC_THRESHOLD,
        )
        self.vector_store = ChromaVectorStore(
            collection_name=config.CHROMA_COLLECTION,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        self.reranker = CrossEncoderReranker(model_name=config.RERANKER_MODEL)
        self.structurer = DocumentStructurer()

        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path: str, verbose: bool = False) -> PipelineResult:
        image_path = str(Path(image_path).resolve())
        logger.info(f"=== Processing: {image_path} ===")

        logger.info("Stage 1: Image Preprocessing")
        preprocessed = self.preprocessor.process(image_path)
        if verbose:
            logger.info(f"  Lines detected: {len(preprocessed.line_images)}")
            logger.info(f"  Skew angle: {preprocessed.skew_angle:.2f}")

        logger.info("Stage 2: Handwriting OCR (Surya)")
        ocr_result = self.ocr_engine.recognize_lines(
            preprocessed.line_images, source_path=image_path
        )
        if verbose:
            logger.info(f"  OCR lines: {len(ocr_result.lines)}")
            logger.info(f"  Avg confidence: {ocr_result.average_confidence:.2f}")

        logger.info("Stage 3: Text Post-processing")
        processed_text = self.text_processor.process(ocr_result.full_text)
        if verbose:
            logger.info(f"  Corrections: {processed_text.corrections_made}")
            logger.info(f"  Word count: {processed_text.word_count}")

        logger.info("Stage 4: Semantic Chunking & Embedding")
        chunks = self.chunker.chunk_and_embed(
            processed_text.cleaned_text,
            source_document=Path(image_path).name,
        )

        logger.info("Stage 4b: Indexing in ChromaDB")
        added = self.vector_store.add_chunks(chunks)
        if verbose:
            logger.info(f"  Chunks indexed: {added}")
            logger.info(f"  Total in store: {self.vector_store.count()}")

        logger.info("Stage 6: Document Structuring")
        title = Path(image_path).stem.replace("_", " ").replace("-", " ").title()
        structured = self.structurer.structure(
            processed_text.cleaned_text,
            title=title,
            source_files=[image_path],
        )

        self._save_outputs(structured, Path(image_path).stem)

        return PipelineResult(
            source_path=image_path,
            ocr_text=ocr_result.full_text,
            cleaned_text=processed_text.cleaned_text,
            ocr_confidence=ocr_result.average_confidence,
            chunk_count=len(chunks),
            structured_document=structured,
        )

    def process_directory(self, directory: str, verbose: bool = False) -> List[PipelineResult]:
        dir_path = Path(directory)
        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
        image_files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in extensions)

        if not image_files:
            logger.warning(f"No images found in {directory}")
            return []

        logger.info(f"Found {len(image_files)} images in {directory}")
        results = []
        for img_file in image_files:
            try:
                result = self.process_image(str(img_file), verbose=verbose)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")

        return results

    def query(self, query_text: str, use_reranking: bool = True, use_hybrid: bool = True) -> QueryResult:
        logger.info(f"Query: {query_text}")

        query_embedding = self.chunker.embed_query(query_text)

        if use_hybrid:
            logger.info("Stage 5a: Hybrid Search (vector + keyword)")
            results = self.vector_store.hybrid_search(
                query_embedding,
                query_text,
                top_k=self.config.TOP_K_RETRIEVAL,
            )
        else:
            logger.info("Stage 5a: Vector Search")
            results = self.vector_store.search(
                query_embedding,
                top_k=self.config.TOP_K_RETRIEVAL,
            )

        reranked = []
        if use_reranking and results:
            logger.info("Stage 5b: Cross-Encoder Re-ranking")
            reranked = self.reranker.rerank(
                query_text,
                results,
                top_k=self.config.TOP_K_RERANK,
            )
            context_chunks = [r.text for r in reranked]
        else:
            context_chunks = [r.text for r in results[: self.config.TOP_K_RERANK]]

        answer_context = "\n\n---\n\n".join(context_chunks)

        return QueryResult(
            query=query_text,
            retrieved_chunks=results,
            reranked_chunks=reranked,
            answer_context=answer_context,
        )

    def _save_outputs(self, structured_doc, base_name: str):
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / f"{base_name}.md"
        md_path.write_text(structured_doc.to_markdown(), encoding="utf-8")
        logger.info(f"  Saved: {md_path}")

        json_path = output_dir / f"{base_name}.json"
        json_path.write_text(structured_doc.to_json(), encoding="utf-8")
        logger.info(f"  Saved: {json_path}")

        txt_path = output_dir / f"{base_name}.txt"
        txt_path.write_text(structured_doc.to_plain_text(), encoding="utf-8")
        logger.info(f"  Saved: {txt_path}")
