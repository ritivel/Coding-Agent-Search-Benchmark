"""Pipeline configuration for Experiment 2."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEMANTIC_THRESHOLD = 0.75

TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5

CHROMA_COLLECTION = "handwritten_docs"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

SURYA_LANGUAGES = ["en"]

IMAGE_PREPROCESSING = {
    "max_width": 2048,
    "denoise_strength": 10,
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),
    "adaptive_block_size": 15,
    "adaptive_c": 11,
    "deskew": True,
    "binarize": True,
}

OUTPUT_DIR = str(BASE_DIR / "output")
SAMPLE_IMAGES_DIR = str(BASE_DIR / "sample_images")
