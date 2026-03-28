"""Configuration for Experiment 0 — Handwritten Document RAG Pipeline."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "output"

TROCR_MODEL = "microsoft/trocr-large-handwritten"
TROCR_PROCESSOR = "microsoft/trocr-large-handwritten"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

VECTOR_STORE_DIR = BASE_DIR / "vector_store_data"

IMAGE_PREPROCESSING = {
    "target_dpi": 300,
    "binarize_threshold": 0,
    "denoise_strength": 10,
    "deskew": True,
    "resize_max_width": 2048,
}

TEXT_POSTPROCESSING = {
    "min_line_confidence": 0.3,
    "merge_hyphenated": True,
    "fix_common_ocr_errors": True,
}

DOCUMENT_STRUCTURE = {
    "detect_headings": True,
    "detect_paragraphs": True,
    "detect_lists": True,
    "output_formats": ["json", "markdown", "txt"],
}

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 5
