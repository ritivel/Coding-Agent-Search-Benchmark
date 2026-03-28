"""
Text post-processing for OCR output cleanup.

Enhanced over Experiment 0 with:
  - Extended OCR substitution table (more known confusions)
  - Sentence boundary repair (capitalise after full stop)
  - Line-break healing heuristic for paragraphs
  - Configurable pipeline stage toggling
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

COMMON_OCR_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"\bl\b(?=[A-Z])", "I"),       # lone "l" before uppercase → "I"
    (r"(?<=[a-z])0(?=[a-z])", "o"),  # zero inside a word → "o"
    (r"(?<=[a-z])1(?=[a-z])", "l"),  # one inside a word → "l"
    (r"\brn\b", "m"),                # "rn" → "m"
    (r"(?<=[a-z])vv(?=[a-z])", "w"), # "vv" → "w"
    (r"\bII\b(?!\s*[.:,])", "II"),   # keep roman numeral II (no-op guard)
    (r"(?<=[a-z])cI(?=[a-z])", "d"), # "cI" → "d"
    (r",,", ","),
    (r"\.\.", "."),
    (r"\?{2,}", "?"),
    (r"!{2,}", "!"),
]


class TextPostProcessor:
    """Clean and normalise raw OCR text."""

    def __init__(
        self,
        merge_hyphenated: bool = True,
        fix_common_errors: bool = True,
        collapse_whitespace: bool = True,
        repair_sentences: bool = True,
    ):
        self.merge_hyphenated = merge_hyphenated
        self.fix_common_errors = fix_common_errors
        self.collapse_whitespace = collapse_whitespace
        self.repair_sentences = repair_sentences

    def normalise_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2014", "—").replace("\u2013", "–")
        text = text.replace("\u00a0", " ")
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

    def _repair_sentence_boundaries(self, text: str) -> str:
        """Capitalise the first letter after sentence-ending punctuation."""
        if not self.repair_sentences:
            return text

        def _cap_match(m: re.Match) -> str:
            return m.group(1) + m.group(2).upper()

        return re.sub(r"([.!?]\s+)([a-z])", _cap_match, text)

    def _heal_line_breaks(self, text: str) -> str:
        """Rejoin lines that are not paragraph breaks.

        Two consecutive newlines → paragraph break (keep).
        Single newline between non-empty lines → space (rejoin).
        """
        lines = text.split("\n")
        healed: list[str] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                healed.append("")
                continue
            if i > 0 and healed and healed[-1] != "":
                healed[-1] = healed[-1].rstrip() + " " + stripped
            else:
                healed.append(stripped)
        return "\n".join(healed)

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
        """Run the full post-processing pipeline."""
        text = self.normalise_unicode(text)
        text = self.merge_hyphenated_words(text)
        text = self.fix_ocr_errors(text)
        text = self.remove_artifacts(text)
        text = self._repair_sentence_boundaries(text)
        text = self.collapse_spaces(text)
        logger.info("Post-processed text: %d chars", len(text))
        return text
