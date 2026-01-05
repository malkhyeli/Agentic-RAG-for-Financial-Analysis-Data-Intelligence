from __future__ import annotations

import json
import sys

from src.app.modes import AppMode, run_routed_query
from src.docqa.pipeline import DocQAConfig, REFUSAL_TEXT, run_docqa
from src.docqa.pipeline import build_evidence_from_pdf_tool


class FakePdfTool:
    def __init__(self, results: dict):
        self._results = results

    def _run(self, query: str) -> str:
        return json.dumps(self._results)


class StubLLM:
    def call(self, prompt: str) -> str:
        if prompt.startswith("Extract atomic, verifiable factual claims"):
            return json.dumps(["Alpha is mentioned."])
        if prompt.startswith("You are a strict claim verifier"):
            return json.dumps(
                {"verdict": "SUPPORTED", "page": 1, "quote": "Alpha", "rationale": "verbatim match"}
            )
        if prompt.startswith("You are a trustworthy Document QA assistant"):
            return "Alpha is mentioned. (p.1)"
        raise AssertionError(f"Unexpected prompt:\n{prompt[:200]}")


def test_docqa_abstains_without_evidence() -> None:
    tool = FakePdfTool({"results": []})
    res = run_docqa("Anything?", pdf_tool=tool, llm=StubLLM(), config=DocQAConfig())
    assert res.answer == REFUSAL_TEXT


def test_docqa_includes_page_citations_when_answered() -> None:
    tool = FakePdfTool(
        {
            "results": [
                {"page": 1, "quote": "Alpha appears in the document. Alpha is important.", "score": 0.95},
                {"page": 2, "quote": "Additional context about Alpha appears here.", "score": 0.75},
                {"page": 3, "quote": "More details are provided about Alpha.", "score": 0.6},
                {"page": 4, "quote": "Alpha is discussed again.", "score": 0.55},
                {"page": 5, "quote": "Alpha is referenced.", "score": 0.5},
            ]
        }
    )

    res = run_docqa("What does it say about Alpha?", pdf_tool=tool, llm=StubLLM(), config=DocQAConfig())
    assert res.answer != REFUSAL_TEXT
    assert "(p.1)" in res.answer
    assert "```text" in res.answer
    assert "\nEvidence\n" in res.answer


def test_docqa_never_invokes_web_tools() -> None:
    sys.modules.pop("src.research.mcp_runner", None)
    tool = FakePdfTool({"results": []})
    _ = run_routed_query(AppMode.DOCUMENT_QA, "Anything?", pdf_tool=tool, llm=StubLLM())
    assert "src.research.mcp_runner" not in sys.modules


def test_build_evidence_keeps_results_without_page() -> None:
    tool = FakePdfTool({"results": [{"quote": "Alpha beta gamma", "score": 0.9}]})
    ev, debug = build_evidence_from_pdf_tool("alpha", tool, top_k=5)
    assert not ev, f"expected fail-closed drop without page, got debug={debug}"
