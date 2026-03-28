"""
Structured output module.

Produces the final digital document in multiple formats:
  - JSON (machine-readable, with full metadata)
  - Markdown (human-readable)
  - Plain text
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import config

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """A logical section of the structured document."""
    heading: str
    content: str
    order: int


@dataclass
class StructuredDocument:
    """The final structured digital document."""
    title: str
    source_image: str
    ocr_engine: str
    ocr_confidence: float
    raw_text: str
    structured_text: str
    sections: list[DocumentSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Export helpers ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]

        lines.append(f"**Source:** `{self.source_image}`  ")
        lines.append(f"**OCR Engine:** {self.ocr_engine}  ")
        lines.append(f"**Confidence:** {self.ocr_confidence:.1f}%  ")
        lines.append(f"**Processed:** {self.created_at}  ")
        lines.append("")
        lines.append("---")
        lines.append("")

        if self.sections:
            for section in self.sections:
                lines.append(f"## {section.heading}")
                lines.append("")
                lines.append(section.content)
                lines.append("")
        else:
            lines.append(self.structured_text)
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("<details><summary>Raw OCR Text</summary>")
        lines.append("")
        lines.append("```")
        lines.append(self.raw_text)
        lines.append("```")
        lines.append("")
        lines.append("</details>")

        return "\n".join(lines)

    def to_plaintext(self) -> str:
        if self.sections:
            parts = []
            for s in self.sections:
                parts.append(f"{s.heading}\n{'=' * len(s.heading)}\n{s.content}\n")
            return "\n".join(parts)
        return self.structured_text


# ── Builder ────────────────────────────────────────────────────────

def parse_sections(structured_text: str) -> list[DocumentSection]:
    """
    Attempt to extract sections from LLM-structured text by looking for
    Markdown headings (## Heading).
    """
    lines = structured_text.split("\n")
    sections: list[DocumentSection] = []
    current_heading = "Content"
    current_content: list[str] = []
    order = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("# "):
            if current_content:
                sections.append(DocumentSection(
                    heading=current_heading,
                    content="\n".join(current_content).strip(),
                    order=order,
                ))
                order += 1
                current_content = []
            current_heading = stripped.lstrip("# ").strip()
        else:
            current_content.append(line)

    if current_content:
        sections.append(DocumentSection(
            heading=current_heading,
            content="\n".join(current_content).strip(),
            order=order,
        ))

    return sections


def build_document(
    title: str,
    source_image: str,
    ocr_engine: str,
    ocr_confidence: float,
    raw_text: str,
    structured_text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> StructuredDocument:
    """Construct a StructuredDocument from pipeline outputs."""
    sections = parse_sections(structured_text)
    metadata = extra_metadata or {}
    metadata["char_count_raw"] = len(raw_text)
    metadata["char_count_structured"] = len(structured_text)
    metadata["section_count"] = len(sections)

    return StructuredDocument(
        title=title,
        source_image=source_image,
        ocr_engine=ocr_engine,
        ocr_confidence=ocr_confidence,
        raw_text=raw_text,
        structured_text=structured_text,
        sections=sections,
        metadata=metadata,
    )


def save_outputs(doc: StructuredDocument, output_dir: Path | None = None) -> dict[str, Path]:
    """Save the structured document in all three formats and return paths."""
    out = output_dir or config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(doc.source_image).stem

    paths = {}

    json_path = out / f"{stem}_structured.json"
    json_path.write_text(doc.to_json(), encoding="utf-8")
    paths["json"] = json_path

    md_path = out / f"{stem}_structured.md"
    md_path.write_text(doc.to_markdown(), encoding="utf-8")
    paths["markdown"] = md_path

    txt_path = out / f"{stem}_structured.txt"
    txt_path.write_text(doc.to_plaintext(), encoding="utf-8")
    paths["plaintext"] = txt_path

    logger.info("Saved outputs to %s: %s", out, list(paths.keys()))
    return paths
