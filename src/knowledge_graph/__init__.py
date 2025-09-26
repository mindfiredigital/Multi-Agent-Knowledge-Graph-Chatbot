"""knowledge_graph public API."""

from importlib.metadata import version, PackageNotFoundError

from .client import KnowledgeGraphClient
from .client_v1 import KnowledgeGraphClientV1
from .documents import (
    DocumentProcessor,
    TxtProcessor,
    DocxProcessor,
    MarkdownProcessor,
    HtmlProcessor,
)

# Alias the current client as V2 for clarity
KnowledgeGraphClientV2 = KnowledgeGraphClient

__all__ = [
    "KnowledgeGraphClient",  # Keep original for backward compatibility
    "KnowledgeGraphClientV1",  # New V1 implementation
    "KnowledgeGraphClientV2",  # Alias for current client
    "DocumentProcessor",
    "TxtProcessor",
    "DocxProcessor",
    "MarkdownProcessor",
    "HtmlProcessor",
    "PdfProcessor",
    "__version__",
]


def __getattr__(name):
    """Lazy import for PdfProcessor to avoid circular imports."""
    if name == "PdfProcessor":
        from .documents.pdf import PdfProcessor

        return PdfProcessor
    raise AttributeError(f"module 'knowledge_graph' has no attribute '{name}'")


try:
    __version__ = version("knowledge_graph")
except PackageNotFoundError:
    __version__ = "0.0.0"
