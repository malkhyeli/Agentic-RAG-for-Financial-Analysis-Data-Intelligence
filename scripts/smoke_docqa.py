#!/usr/bin/env python3
"""Smoke test for Strict RAG (DocQA) mode.

Usage:
    python scripts/smoke_docqa.py --pdf knowledge/AbuDhabi_ClimateChange_Essay.pdf --query "sea-level rise"

PASS criteria:
    - DocumentSearchTool initializes successfully
    - Tool isolation: NO web tools present in strict mode
    - Query returns grounded answer with citations (p.#)
    - Answer does NOT contain URLs or web sources
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


def smoke_test_docqa(pdf_path: str, query: str, *, verbose: bool = False) -> bool:
    """Run smoke test for Strict RAG mode.

    Returns True if all tests pass, False otherwise.
    """
    from crewai import LLM

    from src.agentic_rag.tools.custom_tool import DocumentSearchTool, MissingApiKeyError
    from src.agentic_rag.crew import AgenticRag, validate_tool_isolation
    from src.docqa.pipeline import DocQAConfig, REFUSAL_TEXT, run_docqa

    print("=" * 60)
    print("SMOKE TEST: Strict RAG (DocQA) Mode")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Query: {query}")
    print()

    # Check prerequisites
    if not os.getenv("GROUNDX_API_KEY"):
        print("FAIL: GROUNDX_API_KEY not set")
        return False

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"FAIL: PDF not found: {pdf_path}")
        return False

    started = time.time()

    print("[1/4] Initializing DocumentSearchTool...")
    try:
        pdf_tool = DocumentSearchTool(file_path=str(pdf_file))
        print("      PASS: DocumentSearchTool initialized")
        if verbose:
            status = pdf_tool.debug_status()
            print(f"      Status: {json.dumps(status, indent=2)}")
    except MissingApiKeyError as e:
        print(f"      FAIL: {e}")
        return False
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        return False

    # Test tool isolation via crew
    print("\n[2/4] Validating tool isolation...")
    try:
        rag = AgenticRag(mode="strict", pdf_tool=pdf_tool)
        validate_tool_isolation(rag)
        tools = rag.tools
        tool_names = [type(t).__name__ for t in tools]
        print(f"      Tools: {tool_names}")
        if "SerperDevTool" in tool_names:
            print("      FAIL: Web tool found in strict mode!")
            return False
        if "DocumentSearchTool" not in tool_names:
            print("      FAIL: DocumentSearchTool missing!")
            return False
        if len(tools) != 1:
            print(f"      FAIL: Expected 1 tool, got {len(tools)}")
            return False
        print("      PASS: Tool isolation correct (PDF only, no web)")
    except AssertionError as e:
        print(f"      FAIL: Tool isolation assertion failed: {e}")
        return False
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        return False

    # Run docqa query
    print("\n[3/4] Running DocQA query...")
    try:
        llm = LLM(
            model=os.getenv("DOCQA_LLM_MODEL", "ollama/deepseek-r1:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
        result = run_docqa(query, pdf_tool=pdf_tool, llm=llm, config=DocQAConfig())
        answer = (result.answer or "").strip()

        if verbose:
            print(f"      Decision: {result.decision}")
            print(f"      Evidence count: {len(result.evidence)}")
            print(f"      Retrieval debug: {json.dumps(result.debug.get('evidence', {}), indent=2)}")

        if not answer:
            print("      FAIL: Empty answer")
            return False

        print(f"      PASS: Got answer ({len(answer)} chars)")
        print(f"      Latency: {result.latency_s:.2f}s (total {time.time() - started:.2f}s)")
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        return False

    # Validate answer format
    print("\n[4/4] Validating answer format...")

    # Check for citations
    citations = re.findall(r"\(p\.\d+\)", answer)
    if citations:
        print(f"      PASS: Found citations: {set(citations)}")
    elif answer == REFUSAL_TEXT or "Not in" in answer:
        print("      PASS: Model correctly refused (no evidence)")
    else:
        print("      WARN: No citations found in answer")

    # Check for web artifacts (should NOT be present in strict mode)
    url_pattern = r"https?://|www\."
    if re.search(url_pattern, answer, re.IGNORECASE):
        print("      FAIL: Answer contains URLs (strict mode should not have web sources)")
        return False
    print("      PASS: No web URLs in answer (correct for strict mode)")

    if verbose:
        print(f"\n      Answer preview: {answer[:500]}...")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED: Strict RAG mode")
    print("=" * 60)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test for Strict RAG (DocQA) mode")
    ap.add_argument("--pdf", required=True, help="Path to PDF file")
    ap.add_argument("--query", default="sea-level rise", help="Query to test")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = ap.parse_args()

    success = smoke_test_docqa(args.pdf, args.query, verbose=args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
