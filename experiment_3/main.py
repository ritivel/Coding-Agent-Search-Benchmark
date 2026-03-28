#!/usr/bin/env python3
"""
main.py — CLI entry point for the Handwritten Document RAG Pipeline.

Usage examples:

  # Process a single handwritten image → structured document
  python main.py ingest --image path/to/handwriting.png

  # Process all images in a folder
  python main.py ingest --dir path/to/images/

  # Ask a question about ingested documents (RAG Q&A)
  python main.py query "What is the date mentioned in the letter?"

  # Process image and produce structured output without LLM (OCR-only mode)
  python main.py ingest --image photo.jpg --no-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import config
from preprocessing import preprocess
from ocr_engine import extract_text
from text_processing import clean_text, chunk_text
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from structured_output import build_document, save_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


# ── Helpers ────────────────────────────────────────────────────────

def _collect_images(image: str | None, directory: str | None) -> list[Path]:
    """Resolve a single --image path or all images inside --dir."""
    paths: list[Path] = []
    if image:
        p = Path(image)
        if not p.exists():
            logger.error("Image not found: %s", p)
            sys.exit(1)
        paths.append(p)
    if directory:
        d = Path(directory)
        if not d.is_dir():
            logger.error("Directory not found: %s", d)
            sys.exit(1)
        paths.extend(
            sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        )
    if not paths:
        logger.error("No images provided. Use --image or --dir.")
        sys.exit(1)
    return paths


def _process_single_image(
    image_path: Path,
    vector_manager: VectorStoreManager,
    use_llm: bool = True,
) -> dict:
    """Run the full pipeline on one image and return summary info."""
    logger.info("=" * 60)
    logger.info("Processing: %s", image_path.name)

    # Step 1 — Preprocessing
    logger.info("[1/5] Preprocessing image...")
    cv_img, pil_img = preprocess(image_path)

    # Step 2 — OCR
    logger.info("[2/5] Extracting text (OCR)...")
    ocr_result = extract_text(image_path, preprocessed_cv=cv_img, preprocessed_pil=pil_img)
    logger.info(
        "  → %d lines, confidence %.1f%%, engine=%s",
        len(ocr_result.line_results),
        ocr_result.confidence,
        ocr_result.source,
    )

    # Step 3 — Text cleaning & chunking
    logger.info("[3/5] Cleaning and chunking text...")
    cleaned = clean_text(ocr_result.text)
    chunks = chunk_text(cleaned, source_file=str(image_path.name))
    logger.info("  → %d chunks (size=%d, overlap=%d)", len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    # Step 4 — Embed & store
    logger.info("[4/5] Embedding chunks into vector store...")
    vector_manager.add_chunks(chunks)

    # Step 5 — Structured output
    logger.info("[5/5] Generating structured document...")
    if use_llm:
        rag = RAGPipeline(vector_manager)
        structured_text = rag.structure_document(cleaned)
    else:
        structured_text = cleaned

    doc = build_document(
        title=image_path.stem.replace("_", " ").replace("-", " ").title(),
        source_image=str(image_path),
        ocr_engine=ocr_result.source,
        ocr_confidence=ocr_result.confidence,
        raw_text=ocr_result.text,
        structured_text=structured_text,
        extra_metadata={
            "num_chunks": len(chunks),
            "line_results": ocr_result.line_results,
        },
    )

    output_paths = save_outputs(doc)
    logger.info("  → Outputs: %s", {k: str(v) for k, v in output_paths.items()})

    return {
        "image": str(image_path),
        "lines": len(ocr_result.line_results),
        "confidence": ocr_result.confidence,
        "chunks": len(chunks),
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }


# ── CLI Commands ───────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest command: process images → structured documents."""
    images = _collect_images(args.image, args.dir)
    logger.info("Found %d image(s) to process.", len(images))

    vm = VectorStoreManager()
    results = []
    for img_path in images:
        info = _process_single_image(img_path, vm, use_llm=not args.no_llm)
        results.append(info)

    # Persist vector store for later queries
    vm.save()
    logger.info("Vector store saved to %s", config.VECTOR_STORE_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['image']}")
        print(f"    Lines: {r['lines']}  |  Confidence: {r['confidence']:.1f}%  |  Chunks: {r['chunks']}")
        for fmt, path in r["outputs"].items():
            print(f"    → {fmt}: {path}")
    print("=" * 60)


def cmd_query(args: argparse.Namespace) -> None:
    """Query command: ask questions about ingested documents."""
    vm = VectorStoreManager()
    try:
        vm.load()
    except Exception:
        logger.error(
            "No vector store found. Run 'ingest' first to process documents."
        )
        sys.exit(1)

    rag = RAGPipeline(vm)
    result = rag.query(args.question, k=args.top_k)

    print("\n" + "=" * 60)
    print("QUESTION:", result["question"])
    print("-" * 60)
    print("ANSWER:", result["answer"])
    print("-" * 60)
    print(f"Retrieved {result['num_chunks_retrieved']} chunks:")
    for src in result["sources"]:
        print(f"  • [{src['source']}] chunk #{src['chunk_index']} (score: {src['score']:.4f})")
        print(f"    \"{src['text'][:120]}...\"")
    print("=" * 60)


# ── Argument parser ────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handwritten-rag",
        description="RAG pipeline for handwritten document digitisation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Process handwritten images")
    ingest_p.add_argument("--image", "-i", help="Path to a single image file")
    ingest_p.add_argument("--dir", "-d", help="Path to a directory of images")
    ingest_p.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM structuring (OCR-only mode)",
    )

    # query
    query_p = subparsers.add_parser("query", help="Ask questions about ingested docs")
    query_p.add_argument("question", help="Your question")
    query_p.add_argument("--top-k", type=int, default=config.TOP_K_RESULTS)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
