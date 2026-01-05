#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


FEATURE_NAMES = [
    "top_score",
    "mean_topk",
    "score_gap",
    "std_topk",
    "num_quotes",
    "unique_pages",
    "page_spread",
    "total_quote_chars",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_labels(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(dict(row))
        return out

    if path.suffix.lower() in {".jsonl"}:
        return _load_jsonl(path)

    if path.suffix.lower() in {".json"}:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        raise ValueError("Expected a JSON list of objects.")

    raise ValueError(f"Unsupported label format: {path}")


def _parse_supported(row: dict[str, Any]) -> Optional[int]:
    if "supported" in row:
        v = row["supported"]
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (int, float)):
            return 1 if int(v) == 1 else 0
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"1", "true", "yes", "supported"}:
                return 1
            if t in {"0", "false", "no", "not_supported", "not supported"}:
                return 0
    if "expected_decision" in row:
        t = str(row["expected_decision"]).strip().lower()
        if t in {"supported", "answer"}:
            return 1
        if t in {"not_supported", "not supported", "abstain"}:
            return 0
    return None


def _extract_features(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    debug = entry.get("debug", {}) if isinstance(entry.get("debug"), dict) else {}
    feats = debug.get("features")
    if isinstance(feats, dict):
        return feats
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Train an answerability model from logs + labels.")
    ap.add_argument("--logs", type=Path, default=Path("logs/queries.jsonl"))
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/answerability.pkl"))
    ap.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    args = ap.parse_args()

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise SystemExit(
            f"scikit-learn is required for training but is not installed: {e}\n"
            "Install with: python3 -m pip install scikit-learn"
        )

    if not args.logs.exists():
        raise SystemExit(f"Missing logs file: {args.logs}")
    if not args.labels.exists():
        raise SystemExit(f"Missing labels file: {args.labels}")

    logs = _load_jsonl(args.logs)
    # Index latest features by (query, pdf_hash)
    feature_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for e in logs:
        if e.get("mode") != "document_qa":
            continue
        q = e.get("query")
        if not isinstance(q, str) or not q.strip():
            continue
        h = e.get("pdf_hash")
        h = h if isinstance(h, str) and h.strip() else None
        feats = _extract_features(e)
        if not feats:
            continue
        feature_by_key[(q.strip(), h)] = feats

    labels = _load_labels(args.labels)
    X: list[list[float]] = []
    y: list[int] = []

    for row in labels:
        q = row.get("query")
        if not isinstance(q, str) or not q.strip():
            continue
        pdf_hash = row.get("pdf_hash")
        h = pdf_hash if isinstance(pdf_hash, str) and pdf_hash.strip() else None
        target = _parse_supported(row)
        if target is None:
            continue

        feats = feature_by_key.get((q.strip(), h))
        if feats is None and h is None:
            # Join by query only (best-effort)
            feats = feature_by_key.get((q.strip(), None))
        if feats is None:
            continue

        X.append([float(feats.get(k, 0.0) or 0.0) for k in FEATURE_NAMES])
        y.append(int(target))

    if len(X) < 20:
        raise SystemExit(f"Need >= 20 labeled examples; got {len(X)}")

    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method=args.calibration, cv=5)
    clf.fit(X, y)

    from src.docqa.gate import SklearnAnswerabilityModel

    model = SklearnAnswerabilityModel(feature_names=FEATURE_NAMES, classifier=clf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with args.out.open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {args.out} ({len(X)} samples, calibration={args.calibration})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

