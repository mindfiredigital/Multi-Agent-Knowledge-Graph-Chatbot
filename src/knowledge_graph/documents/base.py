"""Base classes for document processing."""

from abc import ABC, abstractmethod
from typing import List


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    def extract_text(self, content: bytes) -> List[str]:
        """Extract text chunks from document content.

        Args:
            content: Raw document content as bytes

        Returns:
            List of text chunks (e.g., pages, paragraphs, sections)

        Raises:
            ValueError: If the document format is invalid
            IOError: If the document cannot be processed
        """

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (e.g., ['.pdf', '.txt'])
        """

    def validate_content(self, content: bytes) -> bool:
        """Validate that the content matches the expected format.

        Args:
            content: Raw document content as bytes

        Returns:
            True if content appears valid for this processor
        """
        return len(content) > 0
