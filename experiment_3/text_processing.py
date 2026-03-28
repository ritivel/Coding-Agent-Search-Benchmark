"""
Text processing module: cleaning, normalisation, and chunking.

Takes raw OCR output and prepares it for embedding / vector storage.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    index: int
    source_file: str
    start_char: int
    end_char: int


# ── Cleaning ───────────────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    """Normalise to NFC and strip invisible control characters."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def fix_ocr_artefacts(text: str) -> str:
    """Heuristic fixes for common OCR misreads."""
    replacements = {
        "|": "I",
        "0": "O",  # only outside digit contexts — handled carefully below
        "l1": "ll",
        "rn": "m",  # common Tesseract confusion
    }
    for old, new in replacements.items():
        if old in ("0",):
            text = re.sub(r"(?<![0-9])0(?![0-9])", new, text)
        else:
            text = text.replace(old, new)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one; normalise line endings."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text: str, fix_artefacts: bool = False) -> str:
    """Full cleaning pipeline."""
    text = normalise_unicode(raw_text)
    if fix_artefacts:
        text = fix_ocr_artefacts(text)
    text = collapse_whitespace(text)
    return text


# ── Chunking ───────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source_file: str = "unknown",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split cleaned text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter (paragraph → sentence → word boundaries).
    """
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = splitter.create_documents(
        [text],
        metadatas=[{"source": source_file}],
    )

    chunks = []
    offset = 0
    for i, doc in enumerate(documents):
        start = text.find(doc.page_content, offset)
        if start == -1:
            start = offset
        end = start + len(doc.page_content)
        chunks.append(TextChunk(
            text=doc.page_content,
            index=i,
            source_file=source_file,
            start_char=start,
            end_char=end,
        ))
        offset = start + 1

    return chunks
