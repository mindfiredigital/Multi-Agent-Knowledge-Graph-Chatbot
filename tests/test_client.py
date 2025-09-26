import asyncio
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from knowledge_graph.client import KnowledgeGraphClient


class DummyEncoder:
    def encode(self, text: str) -> List[int]:
        # Simple tokenization: 1 token per char (deterministic)
        return list(text)



class DummyAsyncResult:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    async def data(self) -> List[Dict[str, Any]]:
        return self._records


class DummyAsyncSession:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    async def __aenter__(self) -> "DummyAsyncSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    async def run(self, query: str, **params: Any) -> DummyAsyncResult:  # type: ignore[override]
        return DummyAsyncResult(self._records)


class DummyDriver:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    def session(self) -> DummyAsyncSession:
        return DummyAsyncSession(self._records)


class DummyChat:
    class DummyCompletions:
        def __init__(self, answer: str) -> None:
            self._answer = answer

        def create(self, model: str, messages: List[Dict[str, str]], temperature: float):  # type: ignore[no-untyped-def]
            class Choice:
                def __init__(self, content: str) -> None:
                    self.message = types.SimpleNamespace(content=content)

            return types.SimpleNamespace(choices=[Choice(self._answer)])

    def __init__(self, answer: str) -> None:
        self.completions = DummyChat.DummyCompletions(answer)


class DummyOpenAIClient:
    def __init__(self, answer: str) -> None:
        self.chat = DummyChat(answer)


class DummyGraphiti:
    def __init__(self) -> None:
        self.driver = DummyDriver([])
        self._episodes: List[Dict[str, Any]] = []
        self._search_result: Any = {}

    async def build_indices_and_constraints(self) -> None:
        return None

    async def add_episode(self, **kwargs: Any) -> None:
        self._episodes.append(kwargs)

    async def search_(self, question: str, group_ids: List[str]) -> Any:  # noqa: ARG002
        return self._search_result


@pytest.fixture(autouse=True)
def patch_graphiti(monkeypatch: pytest.MonkeyPatch) -> DummyGraphiti:
    dummy = DummyGraphiti()
    monkeypatch.setattr("knowledge_graph.client.Graphiti", lambda *a, **k: dummy)
    return dummy


@pytest.fixture(autouse=True)
def patch_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("knowledge_graph.client.tiktoken.get_encoding", lambda name: DummyEncoder())


@pytest.mark.asyncio
async def test_add_text_group_id_default_and_override(patch_graphiti: DummyGraphiti) -> None:
    client = KnowledgeGraphClient(group_id="default_ns")

    await client.add_text("hello")
    await client.add_text("world", group_id="override_ns")

    assert patch_graphiti._episodes[0]["group_id"] == "default_ns"
    assert patch_graphiti._episodes[1]["group_id"] == "override_ns"


@pytest.mark.asyncio
async def test_merge_chunks_and_ingest_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_graphiti: DummyGraphiti) -> None:
    # Prepare a fake .txt file
    file_path = tmp_path / "doc.txt"
    file_path.write_text("irrelevant on-disk content", encoding="utf-8")

    # Patch TxtProcessor to yield deterministic chunks
    class DummyProcessor:
        def extract_text(self, content: bytes) -> List[str]:  # noqa: ARG002
            return ["abc", "d", "efg", "h"]

    from knowledge_graph import client as client_mod

    # Replace the processor for .txt
    client_mod.KnowledgeGraphClient._DOCUMENT_PROCESSORS[".txt"] = DummyProcessor()  # type: ignore[index]

    c = KnowledgeGraphClient(group_id="g1")
    result = await c.ingest_file(str(file_path))

    # With 1 token per char and max_tokens=1000, all chunks will merge into one
    assert result["chunks"] == 1
    assert patch_graphiti._episodes[-1]["group_id"] == "g1"


@pytest.mark.asyncio
async def test_search_normalization_variants(patch_graphiti: DummyGraphiti) -> None:
    client = KnowledgeGraphClient(group_id="test_group")

    class WithModelDump:
        def model_dump(self) -> Dict[str, Any]:
            return {"ok": 1}

    patch_graphiti._search_result = WithModelDump()
    assert await client.search("q") == {"ok": 1}

    class WithDict:
        def dict(self) -> Dict[str, Any]:  # noqa: A003
            return {"ok": 2}

    patch_graphiti._search_result = WithDict()
    assert await client.search("q") == {"ok": 2}

    patch_graphiti._search_result = {"ok": 3}
    assert await client.search("q") == {"ok": 3}


@pytest.mark.asyncio
async def test_get_answer_and_context_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    client = KnowledgeGraphClient(group_id="test_group")

    # Mock search to return edges with scores and nodes
    async def fake_search(question: str, group_id: str) -> Dict[str, Any]:  # noqa: ARG001
        return {
            "edges": [
                {"source_node_uuid": "n1", "target_node_uuid": "n2", "fact": "f1"},
                {"source_node_uuid": "n3", "target_node_uuid": "n4", "fact": "f2"},
            ],
            "edge_reranker_scores": [0.1, -0.2],
            "nodes": [
                {"uuid": "n1", "summary": "s1"},
                {"uuid": "n2", "summary": "s2"},
                {"uuid": "n3", "summary": "s3"},
            ],
        }

    monkeypatch.setattr(client, "search", fake_search)

    # Mock OpenAI client
    monkeypatch.setattr(client, "_build_llm_client", lambda: DummyOpenAIClient("answer"))

    resp = await client.get_answer("Q?", k=5)
    assert resp["answer"] == "answer"
    # Only first edge has positive score → include its fact and connected nodes
    assert "f1" in "\n".join(resp["context"])  # type: ignore[index]
    assert any("s1" in c or "s2" in c for c in resp["context"])  # type: ignore[index]


@pytest.mark.asyncio
async def test_list_documents_filters_by_group(monkeypatch: pytest.MonkeyPatch) -> None:
    client = KnowledgeGraphClient(group_id="ns")

    records = [
        {
            "filename": "doc1.pdf",
            "chunk_count": 2,
            "first_upload": None,
        },
        {
            "filename": "doc2.txt",
            "chunk_count": 1,
            "first_upload": None,
        },
    ]

    # Patch the underlying driver with canned records
    dummy_graphiti = client._graphiti  # type: ignore[attr-defined]
    dummy_graphiti.driver = DummyDriver(records)  # type: ignore[assignment]

    docs = await client.list_documents()
    assert len(docs) == 2
    assert docs[0]["filename"] == "doc1.pdf"


def test_get_supported_document_types() -> None:
    client = KnowledgeGraphClient()
    types_ = client.get_supported_document_types()
    assert ".pdf" in types_ and ".txt" in types_ and ".md" in types_
