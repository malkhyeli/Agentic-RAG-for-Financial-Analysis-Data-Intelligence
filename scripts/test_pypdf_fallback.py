#!/usr/bin/env python3
"""Quick test to verify pypdf fallback is working for student names query.

Usage:
    python scripts/test_pypdf_fallback.py --pdf /path/to/team_contract.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def run_local_search_test(pdf_path: str, query: str = "Name all the students involved in this contract") -> bool:
    """Test the local pypdf search fallback directly."""
    from src.agentic_rag.tools.custom_tool import DocumentSearchTool

    print("=" * 60)
    print("TEST: pypdf Local Search Fallback")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Query: {query}")
    print()

    # Create tool
    print("[1/3] Creating DocumentSearchTool...")
    try:
        tool = DocumentSearchTool(file_path=pdf_path)
        print(f"      Tool created, bucket_id={tool.bucket_id}")
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        return False

    # Test GroundX search first
    print("\n[2/3] Testing GroundX search...")
    try:
        _, groundx_results = tool._search(query=query, n=5)
        print(f"      GroundX returned {len(groundx_results)} results")
        if groundx_results:
            for i, r in enumerate(groundx_results[:3], 1):
                quote = (r.get("quote") or "")[:60]
                print(f"        [{i}] p.{r.get('page')}: {quote}...")
        else:
            print("      (no results)")
    except Exception as e:
        print(f"      Error: {type(e).__name__}: {e}")
        groundx_results = []

    # Test local fallback
    print("\n[3/3] Testing local pypdf fallback...")
    try:
        local_results = tool._local_pdf_search(query=query, n=5)
        print(f"      Local search returned {len(local_results)} results")
        if local_results:
            for i, r in enumerate(local_results[:5], 1):
                quote = (r.get("quote") or "")[:80].replace('\n', ' ')
                print(f"        [{i}] p.{r.get('page')}: {quote}...")
        else:
            print("      (no results)")
    except Exception as e:
        print(f"      Error: {type(e).__name__}: {e}")
        local_results = []

    # Summary
    print("\n" + "=" * 60)
    if local_results and not groundx_results:
        print("PASS: Local fallback found content that GroundX missed!")
        print("      The student names should now be retrievable.")
    elif local_results and groundx_results:
        print("INFO: Both GroundX and local search found results.")
    elif not local_results and not groundx_results:
        print("FAIL: Neither GroundX nor local search found results.")
        print("      Check if the PDF contains the expected content.")
        return False
    else:
        print("INFO: GroundX found results, local search did not (expected).")

    print("=" * 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test pypdf fallback for PDF search.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--query", type=str, default="Name all the students involved in this contract")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"ERROR: PDF not found: {args.pdf}")
        return 1

    success = run_local_search_test(args.pdf, args.query)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
