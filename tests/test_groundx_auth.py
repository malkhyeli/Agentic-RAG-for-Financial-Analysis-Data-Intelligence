from __future__ import annotations

import json

import pytest
from groundx.core.api_error import ApiError

from src.app.groundx import init_document_search_tool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool, MissingApiKeyError
from src.utils.env import GROUNDX_API_KEY_ENV_VARS


def test_document_search_tool_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GROUNDX_API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(MissingApiKeyError):
        DocumentSearchTool(file_path="dummy.pdf")


def test_init_document_search_tool_handles_401(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import src.agentic_rag.tools.custom_tool as custom_tool

    class _FakeBuckets:
        def create(self, name: str):  # noqa: ARG002
            raise ApiError(status_code=401, body={"message": "Your API key is invalid"})

    class _FakeGroundX:
        def __init__(self, api_key: str):  # noqa: ARG002
            self.buckets = _FakeBuckets()

    monkeypatch.setattr(custom_tool, "GroundX", _FakeGroundX)

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    tool, status = init_document_search_tool(str(pdf_path), api_key_override="abcd1234efgh5678")
    assert tool is None
    assert status.ok is False
    assert status.status_code == 401

    blob = json.dumps(status.as_dict())
    assert "abcd1234efgh5678" not in blob


def test_init_document_search_tool_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agentic_rag.tools.custom_tool as custom_tool

    class _StubTool:
        def __init__(self, file_path: str, api_key=None, **kwargs):  # noqa: ARG002
            self.ready = False
            self.in_progress = True
            self.bucket_id = 123
            self.process_id = "proc"
            self.document_ids = []
            self._has_lookup = True

        def debug_status(self):
            return {
                "bucket_id": self.bucket_id,
                "process_id": self.process_id,
                "document_ids": self.document_ids,
                "ingest_status": "training",
            }

    monkeypatch.setattr(custom_tool, "DocumentSearchTool", _StubTool)
    import src.app.groundx as groundx
    monkeypatch.setattr(groundx, "DocumentSearchTool", _StubTool)

    tool, status = init_document_search_tool("dummy.pdf", api_key_override="k")
    assert tool is None
    assert status.ok is False
    assert status.kind == "in_progress"
    assert "training" in status.message


def test_lookup_document_ids_uses_process_id() -> None:
    import src.agentic_rag.tools.custom_tool as custom_tool

    tool = custom_tool.DocumentSearchTool.model_construct()
    setattr(tool, "_has_lookup", True)
    setattr(tool, "document_ids", [])
    setattr(tool, "document_id", None)
    setattr(tool, "bucket_id", 99)
    setattr(tool, "process_id", "proc-123")

    captured: dict = {}

    def fake_with_retries(func, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs

        class _Lookup:
            documents = [{"document_id": "abc"}]

        return _Lookup()

    setattr(tool, "_with_retries", fake_with_retries)

    class _Docs:
        def lookup(self, *args, **kwargs):  # noqa: ARG002
            captured["lookup_called"] = True
            return None

    class _Client:
        def __init__(self):
            self.documents = _Docs()

    setattr(tool, "client", _Client())

    ids = tool._lookup_document_ids()

    assert ids == ["abc"]
    assert tool.document_id == "abc"
    assert captured["args"] == ("proc-123",)
    assert captured["kwargs"] == {}
