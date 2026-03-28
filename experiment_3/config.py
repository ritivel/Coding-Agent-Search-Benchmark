"""
Configuration for the Handwritten Document RAG Pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "output"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

OUTPUT_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# ── OCR Configuration ──────────────────────────────────────────────
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
TESSERACT_PSM = int(os.getenv("TESSERACT_PSM", "6"))  # 6 = uniform block of text

TROCR_MODEL = os.getenv("TROCR_MODEL", "microsoft/trocr-base-handwritten")
USE_TROCR = os.getenv("USE_TROCR", "true").lower() == "true"

# ── Image Preprocessing ───────────────────────────────────────────
PREPROCESSING_TARGET_DPI = 300
PREPROCESSING_DENOISE_STRENGTH = 10
PREPROCESSING_ADAPTIVE_BLOCK_SIZE = 11
PREPROCESSING_ADAPTIVE_C = 2

# ── Text Processing ───────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Embedding Model ───────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── LLM Configuration ─────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")

# ── RAG Configuration ─────────────────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
