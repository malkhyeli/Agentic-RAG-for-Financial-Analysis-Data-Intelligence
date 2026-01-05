from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class AppMode(str, Enum):
    DOCUMENT_QA = "document_qa"
    DEEP_RESEARCH = "deep_research"

    @classmethod
    def labels(cls) -> dict["AppMode", str]:
        return {
            cls.DOCUMENT_QA: "Document QA (fail-closed)",
            cls.DEEP_RESEARCH: "Deep Research (Web)",
        }


@dataclass(frozen=True)
class RoutedResult:
    mode: AppMode
    answer: str
    debug: dict[str, Any]


def run_routed_query(
    mode: AppMode,
    query: str,
    *,
    pdf_tool: Optional[Any] = None,
    llm: Optional[Any] = None,
) -> RoutedResult:
    """Hard-boundary mode router (no silent PDF+web mixing)."""
    if mode == AppMode.DOCUMENT_QA:
        from src.docqa.pipeline import DocQAConfig, run_docqa

        if pdf_tool is None:
            return RoutedResult(mode=mode, answer="Upload a PDF to use Document QA.", debug={"refusal_reason": "no_pdf"})

        res = run_docqa(query, pdf_tool=pdf_tool, llm=llm, config=DocQAConfig())
        debug = {
            "evidence": [e.__dict__ for e in res.evidence],
            "features": res.features.as_dict(),
            "p_supported": res.p_supported,
            "threshold": res.threshold,
            "decision": res.decision,
            "refusal_reason": res.refusal_reason,
            "claims": res.claims,
            "verifications": [v.__dict__ for v in res.claim_verifications],
            "latency_s": res.latency_s,
        }
        return RoutedResult(mode=mode, answer=res.answer, debug=debug)

    from src.research.mcp_runner import run_mcp_research

    res = run_mcp_research(query)
    if res.citations:
        sources_md = "\n".join([f"- {c.url}" for c in res.citations])
        answer = f"{res.summary}\n\nSources:\n{sources_md}"
    else:
        answer = res.summary

    debug = {"sources": res.sources, "latency_s": res.latency_s, "error": res.error}
    return RoutedResult(mode=mode, answer=answer, debug=debug)
