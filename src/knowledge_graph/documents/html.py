"""HTML document processor."""

from typing import List
import re

from bs4 import BeautifulSoup, Tag

from .base import DocumentProcessor


class HtmlProcessor(DocumentProcessor):
    """Processor for HTML documents."""

    def __init__(self, preserve_structure: bool = False):
        """Initialize HTML processor.

        Args:
            preserve_structure: If True, preserve HTML structure in text chunks
        """
        self.preserve_structure = preserve_structure

    def extract_text(self, content: bytes) -> List[str]:
        """Extract text from HTML content."""
        if not self.validate_content(content):
            raise ValueError("Invalid HTML content")

        try:
            # Decode content
            html_text = content.decode("utf-8", errors="replace")

            # Parse HTML
            soup = BeautifulSoup(html_text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            if self.preserve_structure:
                return self._extract_with_structure(soup)
            return self._extract_plain_text(soup)

        except (UnicodeDecodeError, ValueError, TypeError) as e:
            raise IOError(f"Failed to extract text from HTML file: {e}") from e

    def _extract_with_structure(self, soup: BeautifulSoup) -> List[str]:
        """Extract text while preserving HTML structure."""
        sections = []

        # Extract by semantic sections
        for element in soup.find_all(["article", "section", "div"]):
            if element.get("class") and any(
                cls in ["content", "main", "body"] for cls in element.get("class", [])
            ):
                text = self._get_element_text(element)
                if text.strip():
                    sections.append(text.strip())

        # If no semantic sections found, extract by headers
        if not sections:
            sections = self._extract_by_headers(soup)

        # If still no sections, extract paragraphs
        if not sections:
            sections = self._extract_paragraphs(soup)

        return sections

    def _extract_plain_text(self, soup: BeautifulSoup) -> List[str]:
        """Extract plain text without structure."""
        # Get all text
        text = soup.get_text()

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Group into paragraphs (split by empty lines)
        paragraphs = []
        current_paragraph = []

        for line in lines:
            if line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []

        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        return paragraphs

    def _extract_by_headers(self, soup: BeautifulSoup) -> List[str]:
        """Extract text organized by headers."""
        sections = []
        current_section = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]):
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # New header - save previous section
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                current_section = [element.get_text().strip()]
            else:
                # Content element
                text = element.get_text().strip()
                if text:
                    current_section.append(text)

        # Add the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append(section_text)

        return sections

    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract text by paragraphs."""
        paragraphs = []

        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if text:
                paragraphs.append(text)

        return paragraphs

    def _get_element_text(self, element: Tag) -> str:
        """Get text content from an element."""
        text_parts = []

        for child in element.children:
            if isinstance(child, Tag):
                if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    text_parts.append(f"\n{child.get_text().strip()}\n")
                elif child.name == "p":
                    text_parts.append(child.get_text().strip())
                else:
                    text_parts.append(child.get_text().strip())
            else:
                text = str(child).strip()
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    def get_supported_extensions(self) -> List[str]:
        """Get supported HTML extensions."""
        return [".html", ".htm", ".xhtml"]

    def validate_content(self, content: bytes) -> bool:
        """Validate HTML content."""
        if not content:
            return False
        try:
            html_text = content.decode("utf-8", errors="replace")
            BeautifulSoup(html_text, "html.parser")
            return True
        except (UnicodeDecodeError, ValueError, TypeError):
            return False
