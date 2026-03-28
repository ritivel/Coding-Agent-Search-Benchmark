"""Configuration for Experiment 1 — Advanced Handwritten Document RAG Pipeline.

Key differences from Experiment 0:
  - ChromaDB vector store (persistent, auto-managed)
  - Hybrid search: BM25 (keyword) + vector (semantic) with Reciprocal Rank Fusion
  - Cross-encoder re-ranking for precision
  - Recursive character text splitting with metadata enrichment
  - Morphological-enhanced image preprocessing
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "output"

# --- OCR ---
TROCR_MODEL = "microsoft/trocr-large-handwritten"
TROCR_PROCESSOR = "microsoft/trocr-large-handwritten"

# --- Embeddings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# --- Vector Store (ChromaDB) ---
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "handwritten_docs"

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# --- Retrieval ---
TOP_K_RETRIEVAL = 5
TOP_K_RERANK_CANDIDATES = 20
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_ALPHA = 0.7  # weight for vector search in RRF (1-alpha for BM25)
RRF_K = 60  # standard RRF constant

# --- Image Preprocessing ---
IMAGE_PREPROCESSING = {
    "target_dpi": 300,
    "binarize_threshold": 0,
    "denoise_strength": 10,
    "deskew": True,
    "resize_max_width": 2048,
    "morphological_clean": True,
}

# --- Text Post-processing ---
TEXT_POSTPROCESSING = {
    "min_line_confidence": 0.3,
    "merge_hyphenated": True,
    "fix_common_ocr_errors": True,
}

# --- Document Structuring ---
DOCUMENT_STRUCTURE = {
    "detect_headings": True,
    "detect_paragraphs": True,
    "detect_lists": True,
    "output_formats": ["json", "markdown", "txt"],
}
