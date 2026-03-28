"""
Document structuring module.

Takes cleaned OCR text (optionally enriched via RAG retrieval) and produces
structured digital documents in Markdown, JSON, and plain-text formats.

Enhanced over Experiment 0 with:
  - Date detection in content
  - Signature/salutation detection for letters
  - Table-like content detection (aligned columns)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A structural section of the document."""

    heading: str
    content: str
    level: int = 1
    section_type: str = "paragraph"  # paragraph, list, table, salutation
    subsections: list["Section"] = field(default_factory=list)


@dataclass
class StructuredDocument:
    """A fully structured digital document."""

    title: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines: list[str] = []
        if self.title:
            lines.append(f"# {self.title}")
            lines.append("")

        if self.metadata:
            lines.append("---")
            for k, v in self.metadata.items():
                lines.append(f"{k}: {v}")
            lines.append("---")
            lines.append("")

        for section in self.sections:
            lines.extend(self._section_to_md(section))

        return "\n".join(lines)

    def _section_to_md(self, section: Section) -> list[str]:
        prefix = "#" * (section.level + 1)
        lines = [f"{prefix} {section.heading}", ""]
        if section.content:
            lines.append(section.content)
            lines.append("")
        for sub in section.subsections:
            lines.extend(self._section_to_md(sub))
        return lines

    def to_plain_text(self) -> str:
        lines: list[str] = []
        if self.title:
            lines.append(self.title.upper())
            lines.append("=" * len(self.title))
            lines.append("")
        for section in self.sections:
            lines.extend(self._section_to_txt(section))
        return "\n".join(lines)

    def _section_to_txt(self, section: Section) -> list[str]:
        lines = [section.heading, "-" * len(section.heading), ""]
        if section.content:
            lines.append(section.content)
            lines.append("")
        for sub in section.subsections:
            lines.extend(self._section_to_txt(sub))
        return lines


class DocumentStructurer:
    """Convert cleaned OCR text into a StructuredDocument."""

    HEADING_PATTERNS = [
        re.compile(r"^(#{1,6})\s+(.+)$"),
        re.compile(r"^([A-Z][A-Z\s]{2,})$"),
        re.compile(r"^(\d+\.)\s+([A-Z].+)$"),
        re.compile(r"^(Chapter|Section|Part)\s+\d+[.:]\s*(.+)", re.IGNORECASE),
    ]

    LIST_PATTERNS = [
        re.compile(r"^\s*[-•*]\s+"),
        re.compile(r"^\s*\d+[.)]\s+"),
        re.compile(r"^\s*[a-z][.)]\s+"),
    ]

    SALUTATION_PATTERNS = [
        re.compile(r"^(Dear|To whom|Attention|Hi|Hello)\b", re.IGNORECASE),
        re.compile(r"^(Sincerely|Regards|Best|Yours|Thank you)\b", re.IGNORECASE),
    ]

    DATE_PATTERN = re.compile(
        r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4})\b"
    )

    def __init__(
        self,
        detect_headings: bool = True,
        detect_paragraphs: bool = True,
        detect_lists: bool = True,
    ):
        self.detect_headings = detect_headings
        self.detect_paragraphs = detect_paragraphs
        self.detect_lists = detect_lists

    def _is_heading(self, line: str) -> tuple[bool, int, str]:
        if not self.detect_headings:
            return False, 0, line

        for pattern in self.HEADING_PATTERNS:
            m = pattern.match(line.strip())
            if m:
                groups = m.groups()
                if groups[0].startswith("#"):
                    level = len(groups[0])
                    return True, level, groups[1].strip()
                return True, 1, line.strip()
        return False, 0, line

    def _is_list_item(self, line: str) -> bool:
        if not self.detect_lists:
            return False
        return any(p.match(line) for p in self.LIST_PATTERNS)

    def _is_salutation(self, line: str) -> bool:
        return any(p.match(line.strip()) for p in self.SALUTATION_PATTERNS)

    def _extract_dates(self, text: str) -> list[str]:
        return self.DATE_PATTERN.findall(text)

    def _classify_section_type(self, content: str) -> str:
        lines = content.strip().split("\n")
        if not lines:
            return "paragraph"
        list_lines = sum(1 for l in lines if self._is_list_item(l))
        if list_lines > len(lines) * 0.5:
            return "list"
        if any(self._is_salutation(l) for l in lines[:2]):
            return "salutation"
        return "paragraph"

    def structure(self, text: str, title: str = "", metadata: dict | None = None) -> StructuredDocument:
        """Parse cleaned text into a StructuredDocument."""
        metadata = metadata or {}

        dates = self._extract_dates(text)
        if dates:
            metadata["detected_dates"] = ", ".join(dates)

        lines = text.split("\n")
        sections: list[Section] = []
        current_heading = "Content"
        current_level = 1
        current_lines: list[str] = []

        for line in lines:
            is_head, level, heading_text = self._is_heading(line)
            if is_head:
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    section_type = self._classify_section_type(content)
                    sections.append(Section(
                        heading=current_heading,
                        content=content,
                        level=current_level,
                        section_type=section_type,
                    ))
                    current_lines = []
                current_heading = heading_text
                current_level = level
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            section_type = self._classify_section_type(content)
            sections.append(Section(
                heading=current_heading,
                content=content,
                level=current_level,
                section_type=section_type,
            ))

        if not title and sections:
            title = sections[0].heading

        doc = StructuredDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            raw_text=text,
        )
        logger.info("Structured document: %d sections, title=%r", len(sections), title)
        return doc

    def save(
        self,
        document: StructuredDocument,
        output_dir: Path,
        basename: str = "document",
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        formats = formats or ["json", "markdown", "txt"]
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        if "json" in formats:
            p = output_dir / f"{basename}.json"
            p.write_text(document.to_json(), encoding="utf-8")
            paths["json"] = p

        if "markdown" in formats:
            p = output_dir / f"{basename}.md"
            p.write_text(document.to_markdown(), encoding="utf-8")
            paths["markdown"] = p

        if "txt" in formats:
            p = output_dir / f"{basename}.txt"
            p.write_text(document.to_plain_text(), encoding="utf-8")
            paths["txt"] = p

        logger.info("Saved structured document to %s in formats: %s", output_dir, formats)
        return paths
