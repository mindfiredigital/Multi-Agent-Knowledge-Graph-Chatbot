from io import BytesIO
from typing import Any, Dict, List

import pytest

from knowledge_graph.documents.txt import TxtProcessor
from knowledge_graph.documents.markdown import MarkdownProcessor
from knowledge_graph.documents.html import HtmlProcessor
from knowledge_graph.documents.pdf import PdfProcessor
from knowledge_graph.documents.docx import DocxProcessor
import knowledge_graph as kg


def test_txt_processor_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    tp = TxtProcessor()

    # Force low-confidence detection to trigger fallback encodings
    monkeypatch.setattr(
        "knowledge_graph.documents.txt.chardet.detect",
        lambda b: {"encoding": "utf-16", "confidence": 0.1},
    )

    content = ("Line1\n\nLine2 paragraph\n\nLine3").encode("utf-8")
    chunks = tp.extract_text(content)
    assert len(chunks) >= 1

    assert ".txt" in tp.get_supported_extensions()
    assert tp.validate_content(content) is True

    with pytest.raises(ValueError):
        tp.extract_text(b"")


def test_markdown_processor_paths() -> None:
    mp = MarkdownProcessor()

    md = (
        b"---\n"
        b"title: My Doc\n"
        b"author: Alice\n"
        b"date: 2024-01-01\n"
        b"---\n"
        b"# H1 Title\n\nSome **bold** text and a [link](https://example.com).\n\n"
        b"## H2 Header\n\n- item 1\n- item 2\n\n"
        b"```\ncode block\n```\n"
    )

    sections = mp.extract_text(md)
    assert any("H1 Title" in s for s in sections)

    # Exercise paragraphs path by passing content without headers
    paragraphs = mp.extract_text(b"No headers here\n\nJust text")
    assert any("No headers here" in p for p in paragraphs)

    assert ".md" in mp.get_supported_extensions()
    assert mp.validate_content(b"x") is True

    with pytest.raises(ValueError):
        mp.extract_text(b"")


def test_html_processor_paths() -> None:
    html = (
        "<html><head><title>Page</title><meta name=\"author\" content=\"Bob\"></head>"
        "<body><h1>Header</h1><p>Para1</p><div class=\"content\"><p>Inside</p></div>"
        "<script>ignored()</script></body></html>"
    ).encode("utf-8")

    hp_plain = HtmlProcessor(preserve_structure=False)
    chunks_plain = hp_plain.extract_text(html)
    assert any("Para1" in c for c in chunks_plain)

    hp_struct = HtmlProcessor(preserve_structure=True)
    chunks_struct = hp_struct.extract_text(html)
    assert any("Inside" in c for c in chunks_struct)

    assert ".html" in hp_plain.get_supported_extensions()
    assert hp_plain.validate_content(html) is True

    with pytest.raises(ValueError):
        hp_plain.extract_text(b"")


def test_pdf_processor_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    pp = PdfProcessor()

    class DummyPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class DummyReader:
        def __init__(self, bio: BytesIO) -> None:  # noqa: ARG002
            self.pages = [DummyPage("Page1"), DummyPage("")]
            self.metadata = {"/Title": "T", "/Author": "A", "/CreationDate": "C", "/ModDate": "M"}

    # Patch PdfReader used by module
    monkeypatch.setattr("knowledge_graph.documents.pdf.PdfReader", DummyReader)

    content = b"%PDF-1.4 minimal bytes"
    assert pp.validate_content(content) is True

    chunks = pp.extract_text(content)
    assert chunks == ["Page1"]

    assert ".pdf" in pp.get_supported_extensions()

    with pytest.raises(ValueError):
        pp.extract_text(b"")


def test_docx_processor_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    dp = DocxProcessor()

    class DummyPara:
        def __init__(self, text: str) -> None:
            self.text = text

    class DummyCell:
        def __init__(self, text: str) -> None:
            self.text = text

    class DummyRow:
        def __init__(self) -> None:
            self.cells = [DummyCell("c1"), DummyCell("c2")]

    class DummyTable:
        def __init__(self) -> None:
            self.rows = [DummyRow()]

    class DummyCore:
        def __init__(self) -> None:
            self.title = "Ti"
            self.author = "Au"
            self.created = "Cd"
            self.modified = "Md"
            self.subject = "Sj"
            self.keywords = "Kw"
            self.comments = "Cm"
            self.last_modified_by = "Lm"
            self.revision = 1
            self.version = 1

    class DummyDoc:
        def __init__(self, bio: BytesIO) -> None:  # noqa: ARG002
            self.paragraphs = [DummyPara("p1"), DummyPara("")]
            self.tables = [DummyTable()]
            self.core_properties = DummyCore()

    # Patch Document used by module
    monkeypatch.setattr("knowledge_graph.documents.docx.Document", DummyDoc)

    content = b"PK\x03\x04 minimal docx bytes"
    assert dp.validate_content(content) is True

    chunks = dp.extract_text(content)
    assert "p1" in chunks and any("|" in c for c in chunks)

    assert ".docx" in dp.get_supported_extensions()

    with pytest.raises(ValueError):
        dp.extract_text(b"")


def test_package_init_lazy_import_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # PdfProcessor is exported via __getattr__
    PdfCls = getattr(kg, "PdfProcessor")
    assert PdfCls is not None

    with pytest.raises(AttributeError):
        getattr(kg, "NonExistentClass")
