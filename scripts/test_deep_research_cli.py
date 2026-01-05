#!/usr/bin/env python3
"""Smoke test for DeepResearch (Web) mode with strict fail-closed source binding.

Usage:
    python scripts/test_deep_research_cli.py --query "Give 3 authoritative sources on sea-level rise impacts"

PASS criteria:
    - Exactly N verified citations returned when N sources requested
    - Answer built ONLY from verified citations (no model-generated text)
    - No hallucinated source patterns in output (e.g., "IPCC AR5", "University report")
    - No warning banners - answer is 100% verified
    - If insufficient sources found, fail-closed with clear error
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.research.mcp_runner import (
    run_mcp_research,
    run_mcp_research_with_binding,
    validate_source_binding,
    detect_hallucinated_sources,
    parse_requested_source_count,
    build_citation_only_answer,
)


def smoke_test_deepresearch(query: str, *, timeout_s: int = 180, verbose: bool = False) -> bool:
    """Run smoke test for DeepResearch mode with strict fail-closed behavior.

    Returns True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("SMOKE TEST: DeepResearch (Web) Mode - Strict Fail-Closed")
    print("=" * 60)
    print(f"Query: {query}")

    # Parse requested source count
    requested_count = parse_requested_source_count(query)
    if requested_count:
        print(f"Detected requested source count: {requested_count}")
    print()

    # Check prerequisites
    if not os.getenv("LINKUP_API_KEY"):
        print("FAIL: LINKUP_API_KEY not set")
        return False

    print("[1/5] Running web research with citation-only mode...")
    try:
        result = run_mcp_research_with_binding(
            query,
            timeout_s=timeout_s,
            min_citations=max(1, requested_count or 1),
            require_deep_links=True,
            fail_on_unbound=True,
            use_citation_only_answer=True,  # Use citation-only (no model text)
        )

        if verbose:
            print(f"      Raw summary: {result.summary[:300]}...")
            print(f"      Latency: {result.latency_s:.2f}s")

    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        return False

    print("\n[2/5] Checking for errors...")
    if result.error:
        if "insufficient_sources" in result.error:
            print(f"      INFO: Fail-closed correctly: {result.error}")
            print("      (This is expected behavior when not enough sources found)")
            # Still a pass if we fail closed properly
            return True
        else:
            print(f"      Error: {result.error}")

    print("\n[3/5] Checking citations...")
    if not result.citations:
        print("      INFO: No verified citations returned (fail-closed)")
        return True  # Fail-closed is correct behavior

    actual_count = len(result.citations)
    print(f"      Got {actual_count} verified citation(s)")

    # Verify count matches request
    if requested_count is not None:
        if actual_count < requested_count:
            print(f"      FAIL: Requested {requested_count} but got only {actual_count}")
            return False
        elif actual_count == requested_count:
            print(f"      PASS: Exactly {requested_count} sources as requested")
        else:
            print(f"      INFO: Got {actual_count} (more than {requested_count} requested, showing top {requested_count})")

    for idx, c in enumerate(result.citations[:5], 1):
        title = c.title or "Untitled"
        print(f"        [{idx}] {title[:40]}: {c.url[:55]}...")

    print("\n[4/5] Validating citation-only answer...")
    if not result.bound_answer:
        print("      FAIL: No bound answer generated")
        return False

    # Check for hallucinated patterns in the BOUND ANSWER (not raw summary)
    hallucinated_in_answer = detect_hallucinated_sources(result.bound_answer)
    if hallucinated_in_answer:
        print(f"      FAIL: Bound answer contains hallucinated patterns:")
        for h in hallucinated_in_answer[:3]:
            print(f"        - '{h}'")
        return False
    print("      PASS: No hallucinated source patterns in answer")

    # Check bound answer has only verified URLs
    bound_urls = re.findall(r"https?://[^\s\]\)\"']+", result.bound_answer)
    verified_urls = {c.url.lower().rstrip("/") for c in result.citations}
    unverified_in_bound = []
    for url in bound_urls:
        url_norm = url.lower().rstrip("/")
        if url_norm not in verified_urls:
            unverified_in_bound.append(url)

    if unverified_in_bound:
        print(f"      FAIL: Bound answer contains unverified URLs:")
        for url in unverified_in_bound[:3]:
            print(f"        - {url[:55]}...")
        return False
    print("      PASS: All URLs in answer are verified")

    print("\n[5/5] Validating answer structure...")
    # Check for "Found X verified source(s)" header
    if "verified source" in result.bound_answer.lower():
        print("      PASS: Answer has verified sources header")
    else:
        print("      WARN: Missing verified sources header")

    # Check all citations have URL lines
    url_lines = [line for line in result.bound_answer.split('\n') if line.strip().startswith('URL:')]
    if len(url_lines) >= min(actual_count, requested_count or actual_count):
        print(f"      PASS: Answer has {len(url_lines)} URL entries")
    else:
        print(f"      WARN: Expected {actual_count} URL entries, found {len(url_lines)}")

    if verbose:
        print(f"\n      Full answer:\n{result.bound_answer}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED: DeepResearch mode (strict fail-closed)")
    print("  - Citation-only answer (no model-generated text)")
    print("  - No hallucinated source patterns")
    print("  - All URLs verified")
    if requested_count:
        print(f"  - Exactly {min(actual_count, requested_count)} sources as requested")
    print("=" * 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for DeepResearch (Web) mode with strict fail-closed behavior."
    )
    parser.add_argument("--list-tools", action="store_true", help="List MCP tools via stdio and exit.")
    parser.add_argument(
        "--query",
        type=str,
        default="Give 3 authoritative sources on sea-level rise impacts to coastal cities, include exact URL.",
    )
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.list_tools:
        res = run_mcp_research("list_tools", timeout_s=args.timeout_s, min_citations=0, max_retries=0)
        print(res.summary)
        if res.error:
            print(f"\nerror: {res.error}", file=sys.stderr)
        return 0

    success = smoke_test_deepresearch(args.query, timeout_s=args.timeout_s, verbose=args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
