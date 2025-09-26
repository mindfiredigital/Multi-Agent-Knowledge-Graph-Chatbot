"""Markdown document processor."""

from typing import List
import re

import markdown

from .base import DocumentProcessor


class MarkdownProcessor(DocumentProcessor):
    """Processor for Markdown documents."""

    def __init__(self):
        """Initialize the markdown processor with extensions."""
        self.md = markdown.Markdown(
            extensions=[
                "codehilite",
                "fenced_code",
                "tables",
                "toc",
                "nl2br",
                "attr_list",
            ],
            extension_configs={"codehilite": {"css_class": "highlight"}},
        )

    def extract_text(self, content: bytes) -> List[str]:
        """Extract text from Markdown content."""
        if not self.validate_content(content):
            raise ValueError("Invalid Markdown content")

        try:
            # Decode content
            text = content.decode("utf-8", errors="replace")

            # Split into sections based on headers
            sections = self._split_into_sections(text)

            # If no sections found, split by paragraphs
            if not sections:
                sections = self._split_into_paragraphs(text)

            return sections

        except (UnicodeDecodeError, ValueError, TypeError) as e:
            raise IOError(f"Failed to extract text from Markdown file: {e}") from e

    def _split_into_sections(self, text: str) -> List[str]:
        """Split markdown text into sections based on headers."""
        sections = []

        # Split by headers (lines starting with #)
        header_pattern = r"^(#{1,6}\s+.+)$"
        parts = re.split(header_pattern, text, flags=re.MULTILINE)

        current_section = []
        for _, part in enumerate(parts):
            if re.match(header_pattern, part, flags=re.MULTILINE):
                # This is a header
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                current_section = [part]
            else:
                # This is content
                if part.strip():
                    current_section.append(part.strip())

        # Add the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append(section_text)

        return sections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split markdown text into paragraphs."""
        # Remove markdown syntax for cleaner text
        clean_text = self._remove_markdown_syntax(text)

        # Split by double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]

        return paragraphs

    def _remove_markdown_syntax(self, text: str) -> str:
        """Remove markdown syntax to get plain text."""
        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = re.sub(r"_(.*?)_", r"\1", text)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        return text

    def _extract_frontmatter(self, text: str) -> dict:
        """Extract YAML frontmatter from markdown."""
        frontmatter = {}

        # Check for YAML frontmatter
        if text.startswith("---"):
            try:
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    yaml_content = parts[1].strip()
                    if yaml_content:
                        # Simple YAML parsing for common fields
                        for line in yaml_content.split("\n"):
                            if ":" in line:
                                key, value = line.split(":", 1)
                                key = key.strip()
                                value = value.strip().strip("\"'")
                                frontmatter[key] = value
            except (ValueError, IndexError):
                pass

        return frontmatter

    def get_supported_extensions(self) -> List[str]:
        """Get supported Markdown extensions."""
        return [".md", ".markdown", ".mdown", ".mkdn", ".mkd"]

    def validate_content(self, content: bytes) -> bool:
        """Validate Markdown content."""
        return len(content) > 0
