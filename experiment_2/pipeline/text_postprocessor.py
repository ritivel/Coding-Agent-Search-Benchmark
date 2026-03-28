"""Stage 3: Text Post-processing — clean OCR artifacts and normalise text."""

import re
import unicodedata
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

OCR_CORRECTIONS = [
    (r"\brn\b", "m"),
    (r"\bl\b(?=[a-z])", "1"),
    (r"(?<=[a-z])0(?=[a-z])", "o"),
    (r"\.{3,}", "..."),
    (r'["""]', '"'),
    (r"[''']", "'"),
    (r",,", ","),
    (r"\.\.", "."),
    (r"!!", "!"),
    (r"\?\?", "?"),
]


@dataclass
class PostProcessedText:
    original_text: str
    cleaned_text: str
    corrections_made: int = 0
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.cleaned_text)
        self.word_count = len(self.cleaned_text.split())


class TextPostProcessor:
    def __init__(self, apply_corrections: bool = True, min_line_length: int = 2):
        self.apply_corrections = apply_corrections
        self.min_line_length = min_line_length

    def process(self, raw_text: str) -> PostProcessedText:
        corrections = 0
        text = raw_text

        text = unicodedata.normalize("NFKC", text)

        text = self._merge_hyphenated_words(text)

        if self.apply_corrections:
            text, n = self._apply_ocr_corrections(text)
            corrections += n

        text = self._remove_artifacts(text)
        text = self._normalize_whitespace(text)
        text = self._filter_short_lines(text)

        return PostProcessedText(
            original_text=raw_text,
            cleaned_text=text.strip(),
            corrections_made=corrections,
        )

    def _merge_hyphenated_words(self, text: str) -> str:
        return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    def _apply_ocr_corrections(self, text: str) -> tuple:
        count = 0
        for pattern, replacement in OCR_CORRECTIONS:
            new_text = re.sub(pattern, replacement, text)
            if new_text != text:
                count += 1
            text = new_text
        return text, count

    def _remove_artifacts(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) <= 1 and not stripped.isalnum():
                continue
            stripped = re.sub(r"[^\x20-\x7E\n\t]", "", stripped)
            cleaned.append(stripped)
        return "\n".join(cleaned)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _filter_short_lines(self, text: str) -> str:
        lines = text.split("\n")
        filtered = []
        for line in lines:
            if line.strip() == "":
                filtered.append("")
            elif len(line.strip()) >= self.min_line_length:
                filtered.append(line)
        return "\n".join(filtered)

    def process_lines(self, lines: List[str]) -> PostProcessedText:
        combined = "\n".join(lines)
        return self.process(combined)
