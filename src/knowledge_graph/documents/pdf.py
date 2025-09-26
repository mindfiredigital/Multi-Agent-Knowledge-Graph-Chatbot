"""PDF utilities for the library."""

from io import BytesIO
from typing import List

from pypdf import PdfReader

from .base import DocumentProcessor


def extract_text_from_pdf(content: bytes) -> List[str]:
    """Extract plain text per page from a PDF byte stream.

    Returns a list where each element corresponds to a non-empty page's text.
    """
    reader = PdfReader(BytesIO(content))
    texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            texts.append(text.strip())
    return texts


class PdfProcessor(DocumentProcessor):
    """Processor for PDF documents."""

    def extract_text(self, content: bytes) -> List[str]:
        """Extract text from PDF content."""
        if not self.validate_content(content):
            raise ValueError("Invalid PDF content")

        try:
            return extract_text_from_pdf(content)
        except (IOError, ValueError, TypeError) as e:
            raise IOError(f"Failed to extract text from PDF: {e}") from e

    def get_supported_extensions(self) -> List[str]:
        """Get supported PDF extensions."""
        return [".pdf"]

    def validate_content(self, content: bytes) -> bool:
        """Validate PDF content."""
        if not content:
            return False
        try:
            PdfReader(BytesIO(content))
            return True
        except (IOError, ValueError, TypeError):
            return False
