#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PDF_CACHE_DIRNAME = ".cache_uploaded_pdfs"
PDF_CACHE_META = "last.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_ui_cached_pdf_meta(root: Path) -> tuple[Path, Optional[int], Optional[str]]:
    meta_path = root / PDF_CACHE_DIRNAME / PDF_CACHE_META
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cached UI PDF meta at: {meta_path}")

    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid JSON in: {meta_path}")

    pdf_path = obj.get("path")
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ValueError(f"Missing 'path' in: {meta_path}")

    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"Cached PDF missing: {p}")

    bucket_id = obj.get("bucket_id")
    process_id = obj.get("process_id")

    b = None
    if isinstance(bucket_id, int):
        b = bucket_id
    elif isinstance(bucket_id, str) and bucket_id.strip().isdigit():
        b = int(bucket_id.strip())

    pid = process_id if isinstance(process_id, str) and process_id.strip() else None
    return p, b, pid


def _wait_groundx_complete(pdf_tool, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        status_resp = pdf_tool.client.documents.get_processing_status_by_id(
            process_id=pdf_tool.process_id
        )
        last_status = getattr(getattr(status_resp, "ingest", None), "status", None)
        if last_status == "complete":
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for ingest completion (last status: {last_status})")


def _assert_no_web_artifacts(text: str) -> None:
    t = (text or "").lower()
    banned = ["http", "https", "www", ".com", ".org", "doi"]
    for b in banned:
        assert b not in t, f"Found banned substring {b!r} in output: {text!r}"


def _assert_has_quote_blocks_with_citations(text: str) -> int:
    pattern = re.compile(r"\(p\.(\d+)\)\s*\n```text\n(.+?)\n```", re.DOTALL)
    matches = pattern.findall(text or "")
    assert matches, "Expected at least one verbatim quote block with (p.#) citation."
    for pg, quote in matches:
        assert pg.isdigit() and int(pg) >= 1
        assert isinstance(quote, str) and quote.strip()
    return len(matches)


def main() -> int:
    load_dotenv()
    root = _repo_root()
    sys.path.insert(0, str(root))

    if not os.getenv("GROUNDX_API_KEY"):
        print("GROUNDX_API_KEY is missing; cannot run Document QA verification.", file=sys.stderr)
        return 2

    try:
        from src.agentic_rag.doc_qa_failclosed import run_document_qa_failclosed
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool
    except Exception as e:
        print(f"Import failed: {e}", file=sys.stderr)
        return 2

    try:
        pdf_path, bucket_id, process_id = _load_ui_cached_pdf_meta(root)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    kwargs = {}
    if isinstance(bucket_id, int) and isinstance(process_id, str) and process_id.strip():
        kwargs = {"bucket_id": bucket_id, "process_id": process_id}

    pdf_tool = DocumentSearchTool(file_path=str(pdf_path), **kwargs)
    _wait_groundx_complete(pdf_tool, timeout_s=int(os.getenv("GROUNDX_TIMEOUT_S", "180")))

    q1 = "Quote 2 exact sentences that mention 'sea-level rise'. Cite pages."
    q2 = "What is Abu Dhabi's GDP in 2023? Quote it with (p.#)."
    q3 = "Give 3 reputable sources with URLs about sea-level rise projections to 2100."

    out1 = run_document_qa_failclosed(q1, pdf_tool)
    out2 = run_document_qa_failclosed(q2, pdf_tool)
    out3 = run_document_qa_failclosed(q3, pdf_tool)

    for out in [out1, out2, out3]:
        _assert_no_web_artifacts(out)

    assert out2 == "Not in document.", f"Query 2 expected exact refusal, got: {out2!r}"
    assert out3 == "Not in document.", f"Query 3 expected exact refusal, got: {out3!r}"

    assert out1 != "Not in document.", "Query 1 unexpectedly failed closed."
    blocks = _assert_has_quote_blocks_with_citations(out1)
    assert blocks == 2, f"Query 1 expected exactly 2 quote blocks, got: {blocks}"

    print("OK: Document QA fail-closed verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
