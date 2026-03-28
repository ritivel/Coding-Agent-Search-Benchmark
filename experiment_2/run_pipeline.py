#!/usr/bin/env python3
"""CLI entry point for the Handwritten Document RAG Pipeline (Experiment 2)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2 — Handwritten Document RAG Pipeline (Surya OCR + ChromaDB + Re-ranking)"
    )
    parser.add_argument("--image", type=str, help="Path to a single handwritten image")
    parser.add_argument("--image-dir", type=str, help="Directory of handwritten images")
    parser.add_argument("--query", type=str, help="Query the indexed documents")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample test images")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search (vector-only)")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store before processing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.generate_samples:
        from create_sample_images import create_all_samples
        create_all_samples()
        print("Sample images generated in sample_images/")
        if not args.image and not args.image_dir and not args.query:
            return

    import config as cfg
    from pipeline.rag_pipeline import HandwrittenDocumentRAGPipeline

    pipeline = HandwrittenDocumentRAGPipeline(config=cfg)

    if args.reset:
        pipeline.vector_store.reset()
        print("Vector store reset.")

    if args.image:
        result = pipeline.process_image(args.image, verbose=args.verbose)
        print(f"\n{'='*60}")
        print(f"Processed: {result.source_path}")
        print(f"OCR Confidence: {result.ocr_confidence:.2f}")
        print(f"Chunks indexed: {result.chunk_count}")
        print(f"Output saved to: {cfg.OUTPUT_DIR}/")
        print(f"{'='*60}")
        print(f"\nExtracted text preview:\n{result.cleaned_text[:500]}...")

    elif args.image_dir:
        results = pipeline.process_directory(args.image_dir, verbose=args.verbose)
        print(f"\n{'='*60}")
        print(f"Processed {len(results)} images")
        for r in results:
            print(f"  {Path(r.source_path).name}: confidence={r.ocr_confidence:.2f}, chunks={r.chunk_count}")
        print(f"Total chunks in store: {pipeline.vector_store.count()}")
        print(f"Output saved to: {cfg.OUTPUT_DIR}/")
        print(f"{'='*60}")

    if args.query:
        result = pipeline.query(
            args.query,
            use_reranking=not args.no_rerank,
            use_hybrid=not args.no_hybrid,
        )
        print(f"\n{'='*60}")
        print(f"Query: {result.query}")
        print(f"Retrieved chunks: {len(result.retrieved_chunks)}")
        if result.reranked_chunks:
            print(f"After re-ranking: {len(result.reranked_chunks)}")
            print("\nTop results (re-ranked):")
            for i, r in enumerate(result.reranked_chunks, 1):
                print(f"\n--- Result {i} (score: {r.final_score:.4f}) ---")
                print(r.text[:200])
        else:
            print("\nTop results:")
            for i, r in enumerate(result.retrieved_chunks[:5], 1):
                print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
                print(r.text[:200])
        print(f"\n{'='*60}")

    if not any([args.image, args.image_dir, args.query, args.generate_samples]):
        parser.print_help()


if __name__ == "__main__":
    main()
