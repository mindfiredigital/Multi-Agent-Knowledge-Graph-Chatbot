"""Plain text document processor."""

from typing import List
import re
import chardet

from .base import DocumentProcessor


class TxtProcessor(DocumentProcessor):
    """Processor for plain text documents."""

    def extract_text(self, content: bytes) -> List[str]:
        """Extract text from plain text content."""
        if not self.validate_content(content):
            raise ValueError("Invalid text content")

        try:
            # Detect encoding
            detected = chardet.detect(content)
            encoding = detected.get("encoding", "utf-8")
            confidence = detected.get("confidence", 0)

            # Fallback to common encodings if detection confidence is low
            if confidence < 0.7:
                for fallback_encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        text = content.decode(fallback_encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # Last resort: decode with errors='replace'
                    text = content.decode("utf-8", errors="replace")
            else:
                text = content.decode(encoding, errors="replace")

            # Split into paragraphs separated by one or more blank lines
            paragraphs = [
                block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()
            ]

            # If no paragraphs found, return the whole text as one chunk
            if not paragraphs:
                return [text.strip()] if text.strip() else []

            # Merge paragraphs into chunks of up to 2 paragraphs to reduce fragmentation
            merged: List[str] = []
            for i in range(0, len(paragraphs), 2):
                window = paragraphs[i : i + 2]
                merged.append("\n\n".join(window))

            return merged

        except (UnicodeDecodeError, LookupError, TypeError) as e:
            raise IOError(f"Failed to extract text from plain text file: {e}") from e

    def get_supported_extensions(self) -> List[str]:
        """Get supported text extensions."""
        return [".txt", ".text"]

    def validate_content(self, content: bytes) -> bool:
        """Validate text content."""
        return len(content) > 0
