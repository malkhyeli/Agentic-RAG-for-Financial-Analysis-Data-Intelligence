from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from src.docqa.pipeline import DocQAConfig, run_docqa


@dataclass
class Metrics:
    total: int = 0
    expected_supported: int = 0
    expected_not_supported: int = 0
    answered: int = 0
    abstained: int = 0
    far: int = 0  # answered when should abstain
    frr: int = 0  # abstained when should answer
    claim_support_sum: float = 0.0
    claim_support_n: int = 0


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    raise ValueError("Dataset must be a JSON list (or JSONL objects).")


def _wait_groundx_complete(pdf_tool: DocumentSearchTool, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        status_resp = pdf_tool.client.documents.get_processing_status_by_id(process_id=pdf_tool.process_id)
        last_status = getattr(getattr(status_resp, "ingest", None), "status", None)
        if last_status == "complete":
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for ingest completion (last status: {last_status})")


def _decision_from_answer(answer: str) -> str:
    return "not_supported" if (answer or "").strip() == "Not in document." else "supported"


def _expected_from_row(row: dict[str, Any]) -> Optional[str]:
    v = row.get("expected_decision")
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"supported", "not_supported"}:
            return t
        if t == "answer":
            return "supported"
        if t == "abstain":
            return "not_supported"
    return None


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Offline eval runner for Document QA mode.")
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--timeout-s", type=int, default=int(os.getenv("GROUNDX_TIMEOUT_S", "240")))
    ap.add_argument("--llm-model", default=os.getenv("DOCQA_LLM_MODEL", "ollama/deepseek-r1:7b"))
    ap.add_argument("--llm-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    args = ap.parse_args()

    if not os.getenv("GROUNDX_API_KEY"):
        raise SystemExit("GROUNDX_API_KEY is required for eval.")

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    from crewai import LLM

    llm = LLM(model=args.llm_model, base_url=args.llm_base_url, temperature=0)
    pdf_tool = DocumentSearchTool(file_path=str(args.pdf))
    _wait_groundx_complete(pdf_tool, timeout_s=args.timeout_s)

    dataset = _load_dataset(args.dataset)
    m = Metrics()

    for row in dataset:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        expected = _expected_from_row(row)
        if expected is None:
            continue

        m.total += 1
        if expected == "supported":
            m.expected_supported += 1
        else:
            m.expected_not_supported += 1

        res = run_docqa(
            query,
            pdf_tool=pdf_tool,
            llm=llm,
            config=DocQAConfig(top_k=args.top_k, threshold=args.threshold),
        )
        pred = _decision_from_answer(res.answer)
        if pred == "supported":
            m.answered += 1
        else:
            m.abstained += 1

        if pred == "supported" and expected == "not_supported":
            m.far += 1
        if pred == "not_supported" and expected == "supported":
            m.frr += 1

        if res.claim_verifications:
            support_rate = sum(1 for v in res.claim_verifications if v.verdict == "SUPPORTED") / max(1, len(res.claim_verifications))
            m.claim_support_sum += support_rate
            m.claim_support_n += 1

    far_rate = (m.far / m.expected_not_supported) if m.expected_not_supported else 0.0
    frr_rate = (m.frr / m.expected_supported) if m.expected_supported else 0.0
    claim_support_avg = (m.claim_support_sum / m.claim_support_n) if m.claim_support_n else 0.0

    print(json.dumps(
        {
            "total": m.total,
            "answered": m.answered,
            "abstained": m.abstained,
            "FAR": m.far,
            "FRR": m.frr,
            "FAR_rate": far_rate,
            "FRR_rate": frr_rate,
            "claim_support_avg": claim_support_avg,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

