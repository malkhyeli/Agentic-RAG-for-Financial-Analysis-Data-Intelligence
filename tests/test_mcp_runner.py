from __future__ import annotations

import os
from pathlib import Path

import pytest

import src.research.mcp_runner as mcp_runner


def test_shortener_urls_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_VERIFY_URLS", "0")
    citations = [mcp_runner.Citation(url="https://bit.ly/abc123")]
    verified, reason = mcp_runner._enforce_citation_safety(citations, require_deep_links=True, min_citations=1)
    assert verified == []
    assert reason in {"no_citations", "verification_failed"}


def test_homepages_rejected_when_require_deep_links(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_VERIFY_URLS", "0")
    citations = [mcp_runner.Citation(url="https://www.unesco.org/")]
    verified, _ = mcp_runner._enforce_citation_safety(citations, require_deep_links=True, min_citations=1)
    assert verified == []
    assert mcp_runner._is_homepage_url("https://www.unesco.org/")
    assert mcp_runner._is_homepage_url("https://www.unesco.org/en")


def test_missing_scheme_normalized_to_https() -> None:
    assert mcp_runner._normalize_url("example.com/report.pdf") == "https://example.com/report.pdf"


def test_fallback_returns_no_sources_when_citations_insufficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a fake Repo B dir with a placeholder server file so the runner's preflight passes.
    (tmp_path / "server.py").write_text("# placeholder\n", encoding="utf-8")

    async def fake_call_mcp_stdio(*args, **kwargs):
        return "NO_SOURCES", "mcp_failed: fake"

    def fake_call_repo_b_subprocess(*args, **kwargs):
        # Shorteners must never count as citations; this should force NO_SOURCES.
        return "Sources:\n- https://bit.ly/abc123", None

    monkeypatch.setenv("MCP_VERIFY_URLS", "0")
    monkeypatch.setattr(mcp_runner, "_call_mcp_stdio", fake_call_mcp_stdio)
    monkeypatch.setattr(mcp_runner, "_call_repo_b_subprocess", fake_call_repo_b_subprocess)
    monkeypatch.setattr(mcp_runner, "_repo_b_python", lambda _repo_b: Path(os.environ.get("PYTHON", "python")))

    res = mcp_runner.run_mcp_research(
        "anything",
        repo_b_path=tmp_path,
        timeout_s=5,
        min_citations=1,
        max_retries=0,
    )
    assert res.summary == "NO_SOURCES"
    assert res.citations == []
    assert res.sources == []


def test_url_verification_rejects_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import httpx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"httpx must be installed for this test, got: {e}")

    class DummyResponse:
        def __init__(self, status_code: int, content_type: str = "text/html"):
            self.status_code = status_code
            self.headers = {"content-type": content_type}

    class DummyStreamResponse(DummyResponse):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def head(self, url: str):
            return DummyResponse(404)

        def stream(self, method: str, url: str):
            return DummyStreamResponse(404)

    monkeypatch.setattr(httpx, "Client", DummyClient)
    assert mcp_runner._verify_url_live("https://example.com/report", timeout_s=0.1) is False

