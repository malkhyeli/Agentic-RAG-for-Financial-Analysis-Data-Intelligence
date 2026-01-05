#!/usr/bin/env python3
"""
Unified smoke test for the two-mode agentic RAG application.

Tests:
  1. Tool isolation: Strict mode has NO web tools, DeepResearch has NO PDF tools
  2. Strict RAG mode: Answers from PDF only, fails closed on insufficient evidence
  3. DeepResearch mode: Web research with source binding validation

Usage:
  # Test both modes (requires PDF and all API keys)
  python scripts/smoke_test_all.py --pdf knowledge/AbuDhabi_ClimateChange_Essay.pdf

  # Test strict mode only
  python scripts/smoke_test_all.py --pdf knowledge/AbuDhabi_ClimateChange_Essay.pdf --mode strict

  # Test deepresearch mode only
  python scripts/smoke_test_all.py --mode deepresearch

PASS criteria:
  - Tool isolation: Strict has only PDF tool, DeepResearch has only web tool
  - Strict mode: Citations use (p.#) format, no web URLs in answer
  - DeepResearch mode: All URLs in answer are verified, no hallucinated sources
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _check_env(key: str, required_for: str) -> bool:
    val = os.getenv(key, "").strip()
    if not val:
        print(f"[SKIP] {required_for}: missing {key}")
        return False
    return True


def test_tool_isolation(pdf_path: str | None) -> tuple[bool, str]:
    """Test that tools are correctly isolated by mode."""
    _print_section("TOOL ISOLATION TEST")

    try:
        from src.agentic_rag.crew import AgenticRag, validate_tool_isolation
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool, MissingApiKeyError
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False, f"import_error:{e}"

    errors = []

    # Test strict mode isolation (if we have a PDF)
    if pdf_path and Path(pdf_path).is_file() and os.getenv("GROUNDX_API_KEY"):
        print("[INFO] Testing strict mode tool isolation...")
        try:
            pdf_tool = DocumentSearchTool(file_path=pdf_path)
            rag_strict = AgenticRag(mode="strict", pdf_tool=pdf_tool)
            validate_tool_isolation(rag_strict)
            tools = rag_strict.tools
            tool_names = [type(t).__name__ for t in tools]
            print(f"[INFO] Strict mode tools: {tool_names}")

            if "SerperDevTool" in tool_names:
                errors.append("strict_has_web_tool")
            if "DocumentSearchTool" not in tool_names:
                errors.append("strict_missing_pdf_tool")
            if len(tools) != 1:
                errors.append(f"strict_wrong_tool_count:{len(tools)}")

            if not errors:
                print("[PASS] Strict mode has only PDF tool (no web)")
        except AssertionError as e:
            errors.append(f"strict_isolation_assertion:{e}")
        except MissingApiKeyError:
            print("[SKIP] Strict mode test: missing GROUNDX_API_KEY")
        except Exception as e:
            errors.append(f"strict_error:{e}")
    else:
        print("[SKIP] Strict mode test: no PDF or missing GROUNDX_API_KEY")

    # Test deepresearch mode isolation
    print("[INFO] Testing deepresearch mode tool isolation...")
    try:
        rag_deep = AgenticRag(mode="deepresearch")
        validate_tool_isolation(rag_deep)
        tools = rag_deep.tools
        tool_names = [type(t).__name__ for t in tools]
        print(f"[INFO] DeepResearch mode tools: {tool_names}")

        if "DocumentSearchTool" in tool_names:
            errors.append("deep_has_pdf_tool")
        if "SerperDevTool" not in tool_names:
            errors.append("deep_missing_web_tool")
        if len(tools) != 1:
            errors.append(f"deep_wrong_tool_count:{len(tools)}")

        if "deep_" not in str(errors):
            print("[PASS] DeepResearch mode has only web tool (no PDF)")
    except AssertionError as e:
        errors.append(f"deep_isolation_assertion:{e}")
    except Exception as e:
        errors.append(f"deep_error:{e}")

    if errors:
        print(f"[FAIL] Tool isolation errors: {errors}")
        return False, ",".join(errors)

    print("[PASS] Tool isolation test passed")
    return True, "ok"


def test_strict_mode(pdf_path: str, query: str) -> tuple[bool, str]:
    """Test strict RAG mode: answers only from PDF evidence."""
    _print_section("STRICT RAG MODE TEST")

    if not _check_env("GROUNDX_API_KEY", "Strict mode"):
        return False, "missing_groundx_key"

    if not Path(pdf_path).is_file():
        print(f"[FAIL] PDF not found: {pdf_path}")
        return False, "pdf_not_found"

    try:
        from crewai import LLM
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool, MissingApiKeyError
        from src.docqa.pipeline import DocQAConfig, REFUSAL_TEXT, run_docqa
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False, f"import_error:{e}"

    # Test 1: Index PDF
    print(f"[INFO] Indexing PDF: {pdf_path}")
    started = time.time()
    try:
        tool = DocumentSearchTool(file_path=pdf_path)
    except MissingApiKeyError as e:
        print(f"[FAIL] {e}")
        return False, "missing_groundx_key"
    except Exception as e:
        print(f"[FAIL] Failed to index PDF: {e}")
        return False, f"index_error:{e}"

    status = tool.debug_status()
    print(f"[INFO] Index status: ready={status.get('ready')}, docs={len(status.get('document_ids', []))}")

    # Test 2: Query with expected evidence
    print(f"[INFO] Query: {query}")
    llm = LLM(
        model=os.getenv("DOCQA_LLM_MODEL", "ollama/deepseek-r1:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    try:
        res = run_docqa(query, pdf_tool=tool, llm=llm, config=DocQAConfig())
    except Exception as e:
        print(f"[FAIL] run_docqa error: {e}")
        return False, f"docqa_error:{e}"

    print(f"[INFO] Decision: {res.decision}")
    print(f"[INFO] Evidence count: {len(res.evidence)}")
    print(f"[INFO] P(supported): {res.p_supported:.2f}")
    print(f"[INFO] Latency: {res.latency_s:.2f}s (total {time.time() - started:.2f}s)")

    # Validation checks
    errors = []

    # Check: No web URLs in answer (strict mode should only have PDF citations)
    url_pattern = r"https?://|www\."
    # Only check the actual answer text (before Evidence section)
    answer_main = res.answer.split("Evidence")[0] if "Evidence" in res.answer else res.answer
    if re.search(url_pattern, answer_main, re.IGNORECASE):
        errors.append("web_url_in_answer")

    # Check: Has page citations (if answered)
    if res.decision == "answer":
        if "(p." not in res.answer:
            errors.append("missing_page_citation")
        if "Evidence" not in res.answer:
            errors.append("missing_evidence_section")

    # Check: Fail-closed behavior
    if res.answer == REFUSAL_TEXT:
        print(f"[INFO] Fail-closed correctly (refusal_reason={res.refusal_reason})")
    elif res.decision == "answer":
        print(f"[INFO] Answered with citations")
        citations = re.findall(r"\(p\.\d+\)", res.answer)
        print(f"[INFO] Found citations: {set(citations)}")
        print(f"[INFO] Answer preview: {res.answer[:200]}...")
    else:
        errors.append("unexpected_state")

    if errors:
        print(f"[FAIL] Validation errors: {errors}")
        return False, ",".join(errors)

    print("[PASS] Strict mode test passed")
    return True, "ok"


def test_deepresearch_mode(query: str) -> tuple[bool, str]:
    """Test DeepResearch mode: web research with source binding validation."""
    _print_section("DEEPRESEARCH MODE TEST")

    if not _check_env("LINKUP_API_KEY", "DeepResearch mode"):
        return False, "missing_linkup_key"

    # Check MCP server repo exists
    mcp_repo = Path(__file__).resolve().parents[2] / "Multi-Agent-deep-researcher-mcp-windows-linux"
    if not mcp_repo.exists():
        print(f"[FAIL] MCP repo not found: {mcp_repo}")
        return False, "mcp_repo_missing"

    server_py = mcp_repo / "server.py"
    if not server_py.exists():
        print(f"[FAIL] MCP server not found: {server_py}")
        return False, "mcp_server_missing"

    try:
        from src.research.mcp_runner import (
            run_mcp_research_with_binding,
            detect_hallucinated_sources,
        )
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False, f"import_error:{e}"

    print(f"[INFO] Query: {query}")
    print("[INFO] Running MCP research with source binding (this may take 1-2 minutes)...")
    started = time.time()

    try:
        res = run_mcp_research_with_binding(
            query,
            timeout_s=180,
            min_citations=1,
            fail_on_unbound=False,  # Don't fail, report instead
        )
    except Exception as e:
        print(f"[FAIL] run_mcp_research_with_binding error: {e}")
        return False, f"mcp_error:{e}"

    print(f"[INFO] Latency: {res.latency_s:.2f}s (total {time.time() - started:.2f}s)")
    print(f"[INFO] Citations: {len(res.citations)}")
    print(f"[INFO] Source binding valid: {res.source_binding_valid}")
    if res.error:
        print(f"[INFO] Error: {res.error}")

    # Validation checks
    errors = []

    # Check: Has sources or failed closed
    if res.summary.strip().upper() == "NO_SOURCES":
        if res.citations:
            errors.append("no_sources_but_has_citations")
        else:
            print("[INFO] Fail-closed correctly (no valid sources)")
    elif res.citations:
        print("[INFO] Returned with valid citations:")
        for c in res.citations[:3]:
            title = c.title or "Untitled"
            print(f"  - [{title[:30]}] {c.url[:60]}...")
        if len(res.citations) > 3:
            print(f"  ... and {len(res.citations) - 3} more")
    else:
        errors.append("answer_without_citations")

    # Check: No shortener URLs
    shorteners = {"bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly"}
    for c in res.citations:
        for shortener in shorteners:
            if shortener in c.url.lower():
                errors.append(f"shortener_url:{shortener}")

    # Check: Source binding
    if res.unbound_urls:
        print(f"[WARN] Found {len(res.unbound_urls)} unbound URL(s) (removed from bound answer)")
        for url in res.unbound_urls[:2]:
            print(f"  - {url[:60]}...")

    # Check: Hallucinated sources
    hallucinated = detect_hallucinated_sources(res.summary)
    if hallucinated:
        print(f"[WARN] Detected {len(hallucinated)} hallucinated source pattern(s) (removed from bound answer)")
        for h in hallucinated[:2]:
            print(f"  - '{h}'")

    # Check: Bound answer exists and has verified sources
    if res.bound_answer:
        print(f"[INFO] Bound answer generated ({len(res.bound_answer)} chars)")
        if "Verified Sources:" in res.bound_answer:
            print("[PASS] Bound answer has Verified Sources section")
        else:
            print("[WARN] Bound answer missing Verified Sources section")

        # Verify bound answer has no unverified URLs
        bound_urls = re.findall(r"https?://[^\s\]\)\"']+", res.bound_answer)
        verified_urls_norm = {u.lower().rstrip("/") for u in res.sources}
        for url in bound_urls:
            url_norm = url.lower().rstrip("/")
            if url_norm not in verified_urls_norm:
                errors.append(f"unverified_url_in_bound:{url[:30]}")
    elif res.citations:
        errors.append("no_bound_answer")

    if errors:
        print(f"[FAIL] Validation errors: {errors}")
        return False, ",".join(errors)

    # Success if either got valid citations or failed closed properly
    if res.citations or res.summary.strip().upper() == "NO_SOURCES":
        print("[PASS] DeepResearch mode test passed")
        return True, "ok"

    return False, "unexpected_state"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for two-mode agentic RAG app")
    parser.add_argument("--pdf", type=str, default="knowledge/AbuDhabi_ClimateChange_Essay.pdf",
                        help="PDF file for strict mode test")
    parser.add_argument("--query", type=str, default="sea-level rise",
                        help="Query for strict mode test")
    parser.add_argument("--web-query", type=str, default="What are the latest IPCC findings on sea level rise? Provide sources.",
                        help="Query for DeepResearch mode test")
    parser.add_argument("--mode", choices=["strict", "deepresearch", "all"], default="all",
                        help="Which mode(s) to test")
    args = parser.parse_args()

    results: dict[str, tuple[bool, str]] = {}
    all_passed = True

    # Tool isolation test (always run)
    pdf_for_isolation = args.pdf if args.mode in ("strict", "all") else None
    passed, reason = test_tool_isolation(pdf_for_isolation)
    results["tool_isolation"] = (passed, reason)
    if not passed:
        all_passed = False

    if args.mode in ("strict", "all"):
        passed, reason = test_strict_mode(args.pdf, args.query)
        results["strict"] = (passed, reason)
        if not passed:
            all_passed = False

    if args.mode in ("deepresearch", "all"):
        passed, reason = test_deepresearch_mode(args.web_query)
        results["deepresearch"] = (passed, reason)
        if not passed:
            all_passed = False

    # Summary
    _print_section("SUMMARY")
    for name, (passed, reason) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} ({reason})")

    if all_passed:
        print("\n[SUCCESS] All tests passed")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
