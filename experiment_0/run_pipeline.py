#!/usr/bin/env python3
"""
Entry point for Experiment 0 — Handwritten Document RAG Pipeline.

Usage
-----
    # Process a single image
    python run_pipeline.py --image sample_images/sample_letter.png

    # Process all images in a directory
    python run_pipeline.py --image-dir sample_images/

    # Query indexed documents
    python run_pipeline.py --query "What are the ingredients for the cake?"

    # Generate sample images first, then process them
    python run_pipeline.py --generate-samples --image-dir sample_images/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    IMAGE_PREPROCESSING,
    OUTPUT_DIR,
    SAMPLE_IMAGES_DIR,
    TOP_K_RETRIEVAL,
    TROCR_MODEL,
    VECTOR_STORE_DIR,
)
from pipeline import HandwritingRAGPipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def build_pipeline() -> HandwritingRAGPipeline:
    return HandwritingRAGPipeline(
        trocr_model=TROCR_MODEL,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K_RETRIEVAL,
        preprocessing_config=IMAGE_PREPROCESSING,
        persist_dir=VECTOR_STORE_DIR,
    )


def cmd_process_image(pipeline: HandwritingRAGPipeline, image_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")

    result = pipeline.process_image(
        image_path,
        output_dir=OUTPUT_DIR,
        output_formats=["json", "markdown", "txt"],
    )

    print(f"\n--- OCR Result ({len(result.ocr_result.lines)} lines) ---")
    print(result.cleaned_text[:500])
    if len(result.cleaned_text) > 500:
        print(f"... ({len(result.cleaned_text)} chars total)")

    print(f"\n--- Timings ---")
    for step, elapsed in result.timings.items():
        print(f"  {step:20s}: {elapsed:.2f}s")

    print(f"\n--- Output Files ---")
    for fmt, path in result.output_paths.items():
        print(f"  {fmt:10s}: {path}")

    if result.structured_document:
        print(f"\n--- Structured Document (Markdown) ---")
        md = result.structured_document.to_markdown()
        print(md[:1000])
        if len(md) > 1000:
            print(f"... ({len(md)} chars total)")


def cmd_process_directory(pipeline: HandwritingRAGPipeline, image_dir: Path) -> None:
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)

    if not images:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}")
    for img_path in images:
        cmd_process_image(pipeline, img_path)

    pipeline.save_index()
    print(f"\nVector store saved to {VECTOR_STORE_DIR}")


def cmd_query(pipeline: HandwritingRAGPipeline, query: str) -> None:
    try:
        pipeline.load_index()
    except FileNotFoundError:
        print("No saved index found. Process some images first.")
        return

    results = pipeline.query(query)
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"  [{i}] score={score:.4f}  source={chunk.source}")
        print(f"      {chunk.text[:200]}")
        if len(chunk.text) > 200:
            print(f"      ... ({len(chunk.text)} chars)")
        print()


def cmd_generate_samples() -> None:
    from create_sample_images import main as gen_main
    gen_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 0 — Handwritten Document RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, help="Path to a single handwritten document image")
    parser.add_argument("--image-dir", type=Path, help="Directory of images to process")
    parser.add_argument("--query", type=str, help="Query the indexed documents")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample images")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.generate_samples:
        cmd_generate_samples()

    if not any([args.image, args.image_dir, args.query, args.generate_samples]):
        parser.print_help()
        return

    if args.image or args.image_dir or args.query:
        pipeline = build_pipeline()

        if args.image:
            cmd_process_image(pipeline, args.image)
        if args.image_dir:
            cmd_process_directory(pipeline, args.image_dir)
        if args.query:
            cmd_query(pipeline, args.query)


if __name__ == "__main__":
    main()
