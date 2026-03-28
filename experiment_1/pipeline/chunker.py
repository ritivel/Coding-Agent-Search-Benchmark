"""
Recursive character text splitter with metadata enrichment.

Key difference from Experiment 0's simple word-level chunker:
  - Tries natural separators first (paragraphs → sentences → words)
  - Attaches rich metadata (source, chunk index, char count, word count, timestamp)
  - Configurable separator hierarchy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with rich metadata for downstream retrieval and filtering."""

    text: str
    source: str = ""
    chunk_index: int = 0
    char_count: int = 0
    word_count: int = 0
    created_at: str = ""
    ocr_confidence: float = 0.0

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "created_at": self.created_at,
            "ocr_confidence": self.ocr_confidence,
        }


class RecursiveChunker:
    """Split text using a hierarchy of separators, falling back to smaller ones.

    Mimics LangChain's RecursiveCharacterTextSplitter without the dependency.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)
        return text.split(separator)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits back together up to chunk_size, maintaining overlap."""
        merged: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part)
            sep_len = len(separator) if current_parts else 0

            if current_len + part_len + sep_len > self.chunk_size and current_parts:
                merged.append(separator.join(current_parts))

                # keep overlap: drop parts from the front until under overlap budget
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) + len(separator) > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + len(separator)

                current_parts = overlap_parts
                current_len = sum(len(p) for p in current_parts) + len(separator) * max(0, len(current_parts) - 1)

            current_parts.append(part)
            current_len += part_len + sep_len

        if current_parts:
            merged.append(separator.join(current_parts))

        return merged

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        splits = self._split_by_separator(text, sep)

        final_chunks: list[str] = []
        good_splits: list[str] = []

        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, sep)
                    final_chunks.extend(merged)
                    good_splits = []

                if remaining_seps:
                    sub_chunks = self._recursive_split(s, remaining_seps)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s[:self.chunk_size])

        if good_splits:
            merged = self._merge_splits(good_splits, sep)
            final_chunks.extend(merged)

        return [c for c in final_chunks if c.strip()]

    def split_text(
        self,
        text: str,
        source: str = "",
        ocr_confidence: float = 0.0,
    ) -> list[Chunk]:
        """Split text into enriched Chunk objects."""
        raw_chunks = self._recursive_split(text, self.separators)

        chunks: list[Chunk] = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(Chunk(
                text=chunk_text.strip(),
                source=source,
                chunk_index=i,
                ocr_confidence=ocr_confidence,
            ))

        logger.info(
            "Split text into %d chunks (size=%d, overlap=%d, source=%s)",
            len(chunks), self.chunk_size, self.chunk_overlap, source,
        )
        return chunks
