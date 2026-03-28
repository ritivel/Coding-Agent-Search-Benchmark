"""Stage 6: Document Structuring — detect structure and output in multiple formats."""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    heading: str
    content: str
    level: int = 1
    section_type: str = "paragraph"


@dataclass
class StructuredDocument:
    title: str
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.metadata:
            lines.append("---")
            for k, v in self.metadata.items():
                lines.append(f"**{k}:** {v}")
            lines.extend(["---", ""])

        for section in self.sections:
            prefix = "#" * min(section.level + 1, 6)
            if section.heading:
                lines.append(f"{prefix} {section.heading}")
                lines.append("")
            if section.section_type == "list":
                for item in section.content.split("\n"):
                    item = item.strip()
                    if item:
                        if not item.startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                            item = f"- {item}"
                        lines.append(item)
            else:
                lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "title": self.title,
            "metadata": self.metadata,
            "source_files": self.source_files,
            "created_at": self.created_at,
            "sections": [
                {
                    "heading": s.heading,
                    "content": s.content,
                    "level": s.level,
                    "type": s.section_type,
                }
                for s in self.sections
            ],
            "statistics": {
                "section_count": len(self.sections),
                "total_words": sum(len(s.content.split()) for s in self.sections),
                "total_chars": sum(len(s.content) for s in self.sections),
            },
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_plain_text(self) -> str:
        lines = [self.title, "=" * len(self.title), ""]
        for section in self.sections:
            if section.heading:
                lines.append(section.heading)
                lines.append("-" * len(section.heading))
            lines.append(section.content)
            lines.append("")
        return "\n".join(lines)


class DocumentStructurer:
    HEADING_PATTERNS = [
        (r"^#{1,6}\s+(.+)$", "markdown"),
        (r"^([A-Z][A-Z\s]{3,})$", "allcaps"),
        (r"^(\d+\.?\s+[A-Z].+)$", "numbered"),
        (r"^(Chapter|Section|Part)\s+\d", "chapter"),
        (r"^(Introduction|Conclusion|Summary|Abstract|References)\s*$", "keyword"),
    ]

    LIST_PATTERNS = [
        r"^\s*[-*•]\s+",
        r"^\s*\d+[.)]\s+",
        r"^\s*[a-z][.)]\s+",
        r"^\s*[ivxIVX]+[.)]\s+",
    ]

    def structure(self, text: str, title: str = "Untitled Document", source_files: Optional[List[str]] = None) -> StructuredDocument:
        lines = text.split("\n")
        sections = []
        current_heading = ""
        current_content = []
        current_level = 1
        current_type = "paragraph"

        for line in lines:
            heading_match = self._detect_heading(line)
            if heading_match:
                if current_content or current_heading:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            content="\n".join(current_content).strip(),
                            level=current_level,
                            section_type=current_type,
                        )
                    )
                current_heading = heading_match[0]
                current_level = heading_match[1]
                current_content = []
                current_type = "paragraph"
            elif self._is_list_item(line):
                current_content.append(line)
                current_type = "list"
            elif line.strip():
                current_content.append(line)

        if current_content or current_heading:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    content="\n".join(current_content).strip(),
                    level=current_level,
                    section_type=current_type,
                )
            )

        if not sections and text.strip():
            sections.append(
                DocumentSection(heading="", content=text.strip(), level=1, section_type="paragraph")
            )

        word_count = sum(len(s.content.split()) for s in sections)
        metadata = {
            "word_count": word_count,
            "section_count": len(sections),
            "pipeline": "Experiment 2 — Surya OCR + ChromaDB + Re-ranking",
        }

        return StructuredDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            source_files=source_files or [],
        )

    def _detect_heading(self, line: str) -> Optional[tuple]:
        stripped = line.strip()
        if not stripped:
            return None
        for pattern, ptype in self.HEADING_PATTERNS:
            match = re.match(pattern, stripped)
            if match:
                heading_text = match.group(1) if match.lastindex else stripped
                heading_text = re.sub(r"^#{1,6}\s*", "", heading_text)
                level = 1
                if ptype == "markdown":
                    level = len(re.match(r"^(#{1,6})", stripped).group(1))
                elif ptype == "numbered":
                    level = 2
                elif ptype == "allcaps":
                    level = 1
                return (heading_text.strip(), level)
        return None

    def _is_list_item(self, line: str) -> bool:
        return any(re.match(p, line) for p in self.LIST_PATTERNS)
