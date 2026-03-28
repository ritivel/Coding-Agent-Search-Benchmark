"""
Text post-processing module for OCR output cleanup.

Handles common OCR artefacts, merges hyphenated words, fixes spacing,
and normalises punctuation to improve downstream quality.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

COMMON_OCR_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"\bl\b(?=[A-Z])", "I"),       # lone "l" before uppercase likely "I"
    (r"(?<=[a-z])0(?=[a-z])", "o"),  # zero inside a word → "o"
    (r"(?<=[a-z])1(?=[a-z])", "l"),  # one inside a word → "l"
    (r"\brn\b", "m"),                # "rn" → "m" (classic OCR error)
    (r",,", ","),
    (r"\.\.", "."),
]


class TextPostProcessor:
    """Clean and normalise raw OCR text."""

    def __init__(
        self,
        merge_hyphenated: bool = True,
        fix_common_errors: bool = True,
        collapse_whitespace: bool = True,
    ):
        self.merge_hyphenated = merge_hyphenated
        self.fix_common_errors = fix_common_errors
        self.collapse_whitespace = collapse_whitespace

    def normalise_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2014", "—").replace("\u2013", "–")
        return text

    def merge_hyphenated_words(self, text: str) -> str:
        if not self.merge_hyphenated:
            return text
        return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    def fix_ocr_errors(self, text: str) -> str:
        if not self.fix_common_errors:
            return text
        for pattern, replacement in COMMON_OCR_SUBSTITUTIONS:
            text = re.sub(pattern, replacement, text)
        return text

    def collapse_spaces(self, text: str) -> str:
        if not self.collapse_whitespace:
            return text
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def remove_artifacts(self, text: str) -> str:
        """Remove stray single characters and noise tokens."""
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) <= 1 and not stripped.isalnum():
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def process(self, text: str) -> str:
        """Run the full post-processing pipeline on raw OCR text."""
        text = self.normalise_unicode(text)
        text = self.merge_hyphenated_words(text)
        text = self.fix_ocr_errors(text)
        text = self.remove_artifacts(text)
        text = self.collapse_spaces(text)
        logger.info("Post-processed text: %d chars", len(text))
        return text
