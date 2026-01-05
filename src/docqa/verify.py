from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class ClaimVerification:
    claim: str
    verdict: str  # "SUPPORTED" | "UNSUPPORTED"
    evidence_quote: str | None = None
    evidence_page: int | None = None
    rationale: str | None = None


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\n|\n```$", re.MULTILINE)
_CITATION_RE = re.compile(r"\(p\.\d+\)")


def _call_llm(llm: Any, prompt: str) -> str:
    if llm is None:
        return ""
    call = getattr(llm, "call", None)
    if callable(call):
        out = call(prompt)
        return out if isinstance(out, str) else str(out)
    return ""


def _extract_json(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    s = _CODE_FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def _normalize_claim(c: str) -> str:
    s = (c or "").strip()
    s = _CITATION_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_claims(answer_text: str, llm: Any) -> list[str]:
    answer = (answer_text or "").strip()
    if not answer:
        return []

    prompt = (
        "Extract atomic, verifiable factual claims from the ANSWER.\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Return a JSON array of strings.\n"
        "- Each string is a single short claim (no citations like (p.#)).\n"
        "- Do NOT include headings, bullet markers, or evidence quotes.\n\n"
        f"ANSWER:\n{answer}\n"
    )
    raw = _call_llm(llm, prompt)
    obj = _extract_json(raw)

    claims: list[str] = []
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, str):
                continue
            c = _normalize_claim(item)
            if c:
                claims.append(c)

    # Fallback: treat each sentence as a claim.
    if not claims:
        parts = re.split(r"(?<=[.!?])\s+", _CITATION_RE.sub("", answer))
        for p in parts:
            c = _normalize_claim(p)
            if c:
                claims.append(c)

    deduped: list[str] = []
    seen: set[str] = set()
    for c in claims:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    return deduped[:20]


def _evidence_index(evidence: Sequence[Any]) -> tuple[dict[int, list[str]], list[tuple[int, str]]]:
    by_page: dict[int, list[str]] = {}
    flat: list[tuple[int, str]] = []
    for e in evidence:
        try:
            page = int(getattr(e, "page"))
            quote = str(getattr(e, "quote") or "").strip()
        except Exception:
            continue
        if page < 1 or len(quote) < 5:
            continue
        by_page.setdefault(page, []).append(quote)
        flat.append((page, quote))
    return by_page, flat


def _verify_one_claim(claim: str, evidence: Sequence[Any], llm: Any) -> ClaimVerification:
    claim_norm = (claim or "").strip()
    if not claim_norm:
        return ClaimVerification(claim=claim, verdict="UNSUPPORTED", rationale="empty_claim")

    by_page, flat = _evidence_index(evidence)
    allowed_pages = sorted(by_page.keys())
    ev_lines = "\n".join([f"[p.{pg}] {qt}" for pg, qt in flat])

    prompt = (
        "You are a strict claim verifier.\n"
        "Decide whether the CLAIM is fully supported by the EVIDENCE quotes.\n"
        "Rules:\n"
        "- Output JSON only.\n"
        '- verdict must be exactly "SUPPORTED" or "UNSUPPORTED".\n'
        "- If SUPPORTED, you must provide page (int) and quote (a verbatim snippet from ONE evidence quote).\n"
        "- The quote must appear exactly within the provided evidence for that page.\n"
        f"- Allowed pages: {allowed_pages}\n\n"
        f"CLAIM: {claim_norm}\n\n"
        f"EVIDENCE:\n{ev_lines}\n\n"
        'Return JSON: {"verdict":"SUPPORTED|UNSUPPORTED","page":<int|null>,"quote":"<verbatim|null>","rationale":"<short>"}'
    )

    raw = _call_llm(llm, prompt)
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return ClaimVerification(claim=claim_norm, verdict="UNSUPPORTED", rationale="invalid_json")

    verdict = str(obj.get("verdict") or "").strip().upper()
    page = obj.get("page")
    quote = obj.get("quote")
    rationale = obj.get("rationale")

    if verdict != "SUPPORTED":
        return ClaimVerification(
            claim=claim_norm,
            verdict="UNSUPPORTED",
            rationale=str(rationale) if isinstance(rationale, str) else "unsupported",
        )

    try:
        page_i = int(page)
    except Exception:
        return ClaimVerification(claim=claim_norm, verdict="UNSUPPORTED", rationale="bad_page")

    q = str(quote or "").strip()
    if not q:
        return ClaimVerification(claim=claim_norm, verdict="UNSUPPORTED", rationale="missing_quote")

    ok = False
    for ev_q in by_page.get(page_i, []):
        if q in ev_q:
            ok = True
            break
    if not ok:
        return ClaimVerification(claim=claim_norm, verdict="UNSUPPORTED", rationale="quote_not_in_evidence")

    return ClaimVerification(
        claim=claim_norm,
        verdict="SUPPORTED",
        evidence_quote=q,
        evidence_page=page_i,
        rationale=str(rationale) if isinstance(rationale, str) else None,
    )


def verify_claims(claims: Sequence[str], evidence: Sequence[Any], llm: Any) -> list[ClaimVerification]:
    out: list[ClaimVerification] = []
    for c in claims:
        out.append(_verify_one_claim(c, evidence=evidence, llm=llm))
    return out
