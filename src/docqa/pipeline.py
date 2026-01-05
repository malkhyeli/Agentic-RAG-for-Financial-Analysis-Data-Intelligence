from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.docqa.features import RetrievalFeatures, extract_retrieval_features
from src.docqa.gate import GateDecision, decide_answerability
from src.docqa.verify import ClaimVerification, extract_claims, verify_claims

REFUSAL_TEXT = "Not in the provided documents."


@dataclass(frozen=True)
class Evidence:
    quote: str
    page: int
    score: float
    source: str = "pdf"


@dataclass(frozen=True)
class DocQAConfig:
    top_k: int = 15  # Increased from 5 for better coverage
    threshold: float = 0.5  # Lowered from 0.6 for better recall


@dataclass
class DocQARunResult:
    answer: str
    decision: str  # "answer" | "abstain"
    refusal_reason: Optional[str]
    evidence: list[Evidence]
    features: RetrievalFeatures
    p_supported: float
    threshold: float
    claims: list[str]
    claim_verifications: list[ClaimVerification]
    latency_s: float
    debug: dict[str, Any]


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\n|\n```$", re.MULTILINE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b[^\s]+\.(com|org|net)\b", re.IGNORECASE)
_QUOTE_PHRASE_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]")

_STOPWORDS: set[str] = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "document",
    "explain",
    "exact",
    "for",
    "from",
    "describe",
    "give",
    "how",
    "i",
    "in",
    "include",
    "is",
    "it",
    "its",
    "list",
    "long",
    "me",
    "of",
    "on",
    "one",
    "or",
    "pdf",
    "please",
    "provide",
    "quote",
    "quotes",
    "say",
    "sentence",
    "sentences",
    "short",
    "show",
    "summarize",
    "summary",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "to",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "using",
    "verbatim",
    "direct",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "you",
    "your",
}


def _tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    out: set[str] = set()
    for t in toks:
        if t in _STOPWORDS:
            continue
        if len(t) < 3:
            continue
        out.add(t)
    return out


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("-", " ")).strip()


def _extract_quoted_phrases(query: str) -> list[str]:
    if not isinstance(query, str):
        return []
    phrases = []
    for m in _QUOTE_PHRASE_RE.finditer(query):
        p = (m.group(1) or "").strip()
        if p:
            phrases.append(p)
    return phrases


def _parse_requested_sentence_count(query: str) -> Optional[int]:
    if not isinstance(query, str):
        return None
    q_low = query.lower()
    if "quote" not in q_low or "sentence" not in q_low:
        return None
    m = re.search(r"\b(\d{1,2})\b", q_low)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    if n < 1 or n > 10:
        return None
    return n


def _render_quote_blocks(items: Sequence[Evidence]) -> str:
    blocks: list[str] = []
    for e in items:
        blocks.append(f"(p.{e.page})\n```text\n{e.quote.rstrip()}\n```")
    return "\n\n".join(blocks) if blocks else REFUSAL_TEXT


def _select_quotes_for_query(
    query: str,
    evidence: Sequence[Evidence],
    *,
    gate_passed: bool = False,
    p_supported: float = 0.0,
) -> list[Evidence]:
    """Select the most relevant quotes for a query.

    When gate_passed=True and p_supported is high, we trust the retriever's
    semantic ranking and relax the token overlap requirement. This handles
    cases like "Name the students" where the evidence contains names but
    no overlap with query words like "students" or "name".
    """
    q = query or ""
    q_low = q.lower()

    # Document QA is fail-closed and never returns URLs or external "sources".
    if any(k in q_low for k in [" url", " urls", "http", "https", "www", "doi", "sources", "references"]):
        return []

    must_nums = set(re.findall(r"\b\d{4,}\b", q))
    phrases = _extract_quoted_phrases(q)
    must_phrase_norm = _normalize_for_match(phrases[0]) if phrases else None

    q_tokens = _tokens(q)

    # Detect summary/explanation queries - these don't have specific keywords to match
    # so we should use retriever scores directly instead of requiring token overlap
    summary_patterns = ["summarize", "summary", "explain", "overview", "describe",
                        "what is this", "tell me about", "brief", "main points",
                        "key points", "what does", "about this"]
    is_summary_query = any(p in q_low for p in summary_patterns)

    # Separate evidence by relevance to the specific question
    # This ensures we show RELEVANT evidence, not just high-scoring evidence
    relevant: list[tuple[int, float, int, Evidence]] = []  # Has query term overlap
    supplemental: list[tuple[float, int, Evidence]] = []  # No overlap but high retriever score

    for idx, e in enumerate(evidence):
        qt = e.quote or ""
        qt_norm = _normalize_for_match(qt)
        if must_phrase_norm and must_phrase_norm not in qt_norm:
            continue
        if any(n not in qt for n in must_nums):
            continue
        overlap = len(q_tokens & _tokens(qt)) if q_tokens else 0

        if is_summary_query:
            # For summary queries, use retriever score as primary ranking (no token overlap needed)
            # but still require a minimum relevance score to filter out noise
            if e.score >= 0.3:
                relevant.append((int(e.score * 100), e.score, -idx, e))
        elif overlap > 0:
            # Evidence with query term overlap - prioritize by overlap count, then score
            relevant.append((overlap, e.score, -idx, e))
        elif e.score >= 0.5:
            # High-score evidence without overlap - only use if we don't have enough relevant
            supplemental.append((e.score, -idx, e))

    # Sort by relevance (overlap first, then score)
    relevant.sort(reverse=True)
    supplemental.sort(reverse=True)

    # Build final list: relevant first, then supplemental only if we have < 3 relevant
    ordered = [t[-1] for t in relevant]
    if len(ordered) < 3 and supplemental:
        # Add max 2 supplemental evidence pieces
        ordered.extend([t[-1] for t in supplemental[:2]])

    n_sent = _parse_requested_sentence_count(q)
    if n_sent is not None:
        # Return available quotes even if fewer than requested (partial is better than nothing)
        return ordered[:n_sent] if ordered else []

    # Show fewer, more relevant evidence (reduced from 8/5)
    max_quotes = 5 if gate_passed else 3
    return ordered[:max_quotes]


def _parse_tool_results(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    s = _CODE_FENCE_RE.sub("", raw.strip()).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    results = obj.get("results", [])
    return results if isinstance(results, list) else []


def _coerce_score(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return 0.0
    return 0.0


def _coerce_page(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    out: list[str] = []
    for ln in (text or "").splitlines():
        ln2 = ln.strip()
        if not ln2:
            continue
        s = re.sub(r"\s+", " ", ln2).strip()
        if not s:
            continue

        # For short, table-like rows (often names/titles), preserve the line to avoid
        # splitting on middle initials like "P." / "R.".
        looks_like_row = (
            len(s) <= 140
            and (
                re.search(r"\b\d{1,3}\b", s)  # ages, counts, table rows
                or re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b", s)  # names
            )
        )
        if looks_like_row:
            out.append(s)
            continue

        parts = re.split(r"(?<=[.!?])\s+", s)
        out.extend([p.strip() for p in parts if p.strip()])

    return out


def _looks_like_bibliography(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    ref_like = 0
    for ln in lines:
        ln_low = ln.lower()
        if _URL_RE.search(ln) or _DOMAIN_RE.search(ln):
            ref_like += 1
            continue
        if "doi" in ln_low or "retrieved from" in ln_low or "accessed" in ln_low:
            ref_like += 1
            continue
        if "et al" in ln_low:
            ref_like += 1
            continue
        if re.search(r"\b(19|20)\d{2}\b", ln) and ("," in ln or ";" in ln):
            ref_like += 1
            continue
    return (ref_like / max(1, len(lines))) >= 0.7  # Increased from 0.5 to reduce false positives


def _contains_disallowed_patterns(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _URL_RE.search(t) or _DOMAIN_RE.search(t):
        return True
    t_low = t.lower()
    # Only reject citation markers at start of text (not "The source of...")
    start_disallowed = ["source:", "sources:", "reference:", "references:"]
    if any(t_low.startswith(s) for s in start_disallowed):
        return True
    # These patterns indicate external references regardless of position
    anywhere_disallowed = ["retrieved from", "accessed ", "doi:"]
    return any(s in t_low for s in anywhere_disallowed)


def _sanitize_evidence(evidence: list[Evidence]) -> tuple[list[Evidence], dict[str, Any]]:
    accepted: list[Evidence] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for e in evidence:
        if _looks_like_bibliography(e.quote):
            rejected.append({"page": e.page, "quote": e.quote, "reason": "bibliography_like"})
            continue

        # First try to keep the entire quote intact (preserves coherent text)
        full_quote = e.quote.strip()
        if len(full_quote) >= 3 and not _contains_disallowed_patterns(full_quote):
            key = (e.page, full_quote)
            if key not in seen:
                seen.add(key)
                accepted.append(Evidence(quote=full_quote, page=e.page, score=e.score, source=e.source))
            continue

        # Fallback: split into sentences and validate individually
        for sent in _sentences(e.quote):
            s = sent.strip()
            if len(s) < 3:  # Allow very short content like names (e.g., "Bob" = 3 chars)
                continue
            if _contains_disallowed_patterns(s):
                rejected.append({"page": e.page, "quote": s, "reason": "disallowed_pattern"})
                continue

            key = (e.page, s)
            if key in seen:
                continue
            seen.add(key)
            accepted.append(Evidence(quote=s, page=e.page, score=e.score, source=e.source))

    debug = {
        "accepted": [{"page": e.page, "score": e.score, "quote": e.quote} for e in accepted],
        "rejected": rejected,
    }
    return accepted, debug


def build_evidence_from_pdf_tool(query: str, pdf_tool: Any, *, top_k: int) -> tuple[list[Evidence], dict[str, Any]]:
    raw = None
    try:
        raw = pdf_tool._run(query)
    except Exception as e:
        return [], {"tool_error": str(e), "raw": None}

    if isinstance(raw, str):
        # GroundX may return a plain status message, but the document itself can contain the
        # word "processed" (e.g., "recorded, processed, summarized..."), so only match
        # the exact status prefix.
        s = raw.strip().lower()
        if s.startswith("document is still being processed"):
            return [], {"tool_error": "document_processing", "raw": raw}

    results = _parse_tool_results(raw)
    evidence: list[Evidence] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        quote = (r.get("quote") or "").strip()
        if len(quote) < 3:  # Allow very short content like names (e.g., "Bob" = 3 chars)
            continue
        page = _coerce_page(r.get("page"))
        if page is None or page < 1:
            # Fail-closed: if the retriever cannot provide a real page number, do not use it as evidence.
            continue
        score = _coerce_score(r.get("score"))
        evidence.append(Evidence(quote=quote, page=page, score=score, source="pdf"))

    evidence.sort(key=lambda e: e.score, reverse=True)
    sanitized, sanitize_debug = _sanitize_evidence(evidence)
    sanitized.sort(key=lambda e: e.score, reverse=True)
    top = sanitized[: max(0, int(top_k))]
    debug = {
        "raw": raw,
        "num_results": len(results),
        "num_evidence_raw": len(evidence),
        "num_evidence_sanitized": len(sanitized),
        "raw_top_scores": [getattr(e, "score", None) for e in evidence[:5]],
        "sanitize": sanitize_debug,
    }
    return top, debug


def _extract_cited_pages(text: str) -> list[int]:
    if not isinstance(text, str):
        return []
    return [int(x) for x in re.findall(r"\(p\.(\d+)\)", text)]


def _has_any_citation(text: str) -> bool:
    return bool(re.search(r"\(p\.\d+\)", text or ""))


def _contains_web_artifacts(text: str) -> bool:
    return _contains_disallowed_patterns(text or "")


def _validate_answer_text(answer_text: str, *, allowed_pages: set[int]) -> tuple[bool, str]:
    a = (answer_text or "").strip()
    if not a:
        return False, "empty_answer"
    if a == REFUSAL_TEXT:
        return False, "model_refused"
    if _contains_web_artifacts(a):
        return False, "web_artifact"
    cited = _extract_cited_pages(a)
    # Allow short answers (< 30 chars) without citations - these are often names, yes/no, etc.
    if not cited and len(a) >= 30:
        return False, "missing_citations"
    if cited and any(p not in allowed_pages for p in cited):
        return False, "cites_unknown_page"
    return True, "ok"


def _render_evidence_section(evidence: list[Evidence]) -> str:
    lines: list[str] = ["Evidence"]
    for e in evidence:
        lines.append(f"- (p.{e.page})\n  ```text\n  {e.quote.rstrip()}\n  ```")
    return "\n".join(lines)


def _call_llm(llm: Any, prompt: str) -> str:
    if llm is None:
        return ""
    call = getattr(llm, "call", None)
    if callable(call):
        out = call(prompt)
        return out if isinstance(out, str) else str(out)
    return ""


def _answer_prompt(query: str, evidence: Sequence[Evidence]) -> str:
    pages = sorted({e.page for e in evidence})
    ev_lines = []
    for e in evidence:
        ev_lines.append(f"[p.{e.page}] {e.quote}")
    ev_block = "\n".join(ev_lines)
    return (
        "You are a trustworthy Document QA assistant.\n"
        "You must answer ONLY using the provided verbatim evidence quotes from the PDF.\n"
        "Hard rules:\n"
        f"- Allowed citation pages: {pages}\n"
        "- Every sentence must include at least one citation like (p.#) using ONLY allowed pages.\n"
        "- Do NOT include URLs, DOIs, or a Sources/References section.\n"
        f'- If the evidence is insufficient to answer, output exactly: {REFUSAL_TEXT}\n\n'
        f"User query: {query}\n\n"
        "Evidence (verbatim):\n"
        f"{ev_block}\n\n"
        "Return ONLY the answer text (no Evidence section, no Confidence line)."
    )


def _resynth_prompt(query: str, supported_claims: Sequence[str], evidence: Sequence[Evidence]) -> str:
    pages = sorted({e.page for e in evidence})
    claims_block = "\n".join([f"- {c}" for c in supported_claims])
    ev_lines = "\n".join([f"[p.{e.page}] {e.quote}" for e in evidence])
    return (
        "You are a trustworthy Document QA assistant.\n"
        "Rewrite the answer using ONLY the SUPPORTED_CLAIMS.\n"
        "Hard rules:\n"
        f"- Allowed citation pages: {pages}\n"
        "- Every sentence must include at least one citation like (p.#) using ONLY allowed pages.\n"
        "- Do NOT add new claims.\n"
        "- Do NOT include URLs, DOIs, or a Sources/References section.\n\n"
        f"User query: {query}\n\n"
        f"SUPPORTED_CLAIMS:\n{claims_block}\n\n"
        f"EVIDENCE (verbatim):\n{ev_lines}\n\n"
        "Return ONLY the answer text (no Evidence section, no Confidence line)."
    )


def _format_confidence_line(p_supported: float, *, model: str) -> str:
    p = 0.0 if not math.isfinite(p_supported) else max(0.0, min(1.0, float(p_supported)))
    return f"Confidence: P(supported)={p:.2f} ({model})"


def run_docqa(
    query: str,
    *,
    pdf_tool: Any,
    llm: Any = None,
    config: DocQAConfig = DocQAConfig(),
) -> DocQARunResult:
    """Fail-closed Document QA runner.

    Phase 2: retrieval features + baseline statistical gate + grounded answer format.
    """
    started = time.time()
    evidence, ev_debug = build_evidence_from_pdf_tool(query, pdf_tool, top_k=config.top_k)
    features = extract_retrieval_features(evidence)

    if not evidence:
        return DocQARunResult(
            answer=REFUSAL_TEXT,
            decision="abstain",
            refusal_reason="no_evidence",
            evidence=[],
            features=features,
            p_supported=0.0,
            threshold=config.threshold,
            claims=[],
            claim_verifications=[],
            latency_s=time.time() - started,
            debug={"evidence": ev_debug, "retrieval_count": ev_debug.get("num_results", 0)},
        )

    gate: GateDecision = decide_answerability(features, threshold=config.threshold)
    if not gate.supported:
        return DocQARunResult(
            answer=REFUSAL_TEXT,
            decision="abstain",
            refusal_reason=gate.reason,
            evidence=evidence,
            features=features,
            p_supported=gate.p_supported,
            threshold=gate.threshold,
            claims=[],
            claim_verifications=[],
            latency_s=time.time() - started,
            debug={"evidence": ev_debug, "gate": gate.__dict__, "retrieval_count": ev_debug.get("num_results", 0)},
        )

    picked = _select_quotes_for_query(
        query, evidence, gate_passed=gate.supported, p_supported=gate.p_supported
    )
    if not picked:
        return DocQARunResult(
            answer=REFUSAL_TEXT,
            decision="abstain",
            refusal_reason="no_supported_quote",
            evidence=evidence,
            features=features,
            p_supported=gate.p_supported,
            threshold=gate.threshold,
            claims=[],
            claim_verifications=[],
            latency_s=time.time() - started,
            debug={"evidence": ev_debug, "gate": gate.__dict__, "retrieval_count": ev_debug.get("num_results", 0)},
        )

    allowed_pages = {e.page for e in evidence}

    # If no LLM is provided (or the call fails), fall back to quote blocks (still grounded).
    if llm is None:
        draft = _render_quote_blocks(picked)
        evidence_md = _render_evidence_section(evidence)
        final = "\n\n".join([draft, evidence_md])
        return DocQARunResult(
            answer=final,
            decision="answer",
            refusal_reason=None,
            evidence=evidence,
            features=features,
            p_supported=gate.p_supported,
            threshold=config.threshold,
            claims=[],
            claim_verifications=[],
            latency_s=time.time() - started,
            debug={
                "evidence": ev_debug,
                "gate": gate.__dict__,
                "draft": draft,
                "retrieval_count": ev_debug.get("num_results", 0),
                "llm_used": False,
            },
        )

    answer_prompt = _answer_prompt(query, evidence)
    answer_raw = ""
    try:
        answer_raw = _call_llm(llm, answer_prompt)
    except Exception as e:
        draft = _render_quote_blocks(picked)
        evidence_md = _render_evidence_section(evidence)
        final = "\n\n".join([draft, evidence_md])
        return DocQARunResult(
            answer=final,
            decision="answer",
            refusal_reason=None,
            evidence=evidence,
            features=features,
            p_supported=gate.p_supported,
            threshold=config.threshold,
            claims=[],
            claim_verifications=[],
            latency_s=time.time() - started,
            debug={
                "evidence": ev_debug,
                "gate": gate.__dict__,
                "draft": draft,
                "retrieval_count": ev_debug.get("num_results", 0),
                "llm_used": True,
                "llm_error": str(e),
            },
        )

    answer = _CODE_FENCE_RE.sub("", (answer_raw or "").strip()).strip()
    ok, reason = _validate_answer_text(answer, allowed_pages=allowed_pages)
    if not ok:
        # One repair attempt: tell the model exactly what failed.
        repair_prompt = (
            "Your previous answer violated the formatting/grounding rules.\n"
            f"Violation: {reason}\n\n"
            "Rewrite the answer using ONLY the provided evidence.\n"
            "- Every sentence MUST include a citation like (p.#).\n"
            "- Use ONLY allowed pages.\n"
            "- Do NOT include URLs, DOIs, or Sources/References.\n"
            f'- If the evidence is insufficient, output exactly: {REFUSAL_TEXT}\n\n'
            f"ALLOWED_PAGES: {sorted(allowed_pages)}\n\n"
            f"USER_QUERY: {query}\n\n"
            f"EVIDENCE:\n" + "\n".join([f"[p.{e.page}] {e.quote}" for e in evidence]) + "\n\n"
            f"PREVIOUS_ANSWER:\n{answer_raw}\n"
        )
        try:
            answer2_raw = _call_llm(llm, repair_prompt)
        except Exception as e:
            return DocQARunResult(
                answer=REFUSAL_TEXT,
                decision="abstain",
                refusal_reason="llm_error",
                evidence=evidence,
                features=features,
                p_supported=gate.p_supported,
                threshold=config.threshold,
                claims=[],
                claim_verifications=[],
                latency_s=time.time() - started,
                debug={
                    "evidence": ev_debug,
                    "gate": gate.__dict__,
                    "draft": answer_raw,
                    "repair_error": str(e),
                    "answer_validation": {"ok": False, "reason": reason},
                    "retrieval_count": ev_debug.get("num_results", 0),
                },
            )

        answer2 = _CODE_FENCE_RE.sub("", (answer2_raw or "").strip()).strip()
        ok2, reason2 = _validate_answer_text(answer2, allowed_pages=allowed_pages)
        if not ok2:
            return DocQARunResult(
                answer=REFUSAL_TEXT,
                decision="abstain",
                refusal_reason=f"answer_invalid:{reason2}",
                evidence=evidence,
                features=features,
                p_supported=gate.p_supported,
                threshold=config.threshold,
                claims=[],
                claim_verifications=[],
                latency_s=time.time() - started,
                debug={
                    "evidence": ev_debug,
                    "gate": gate.__dict__,
                    "draft": answer_raw,
                    "repair_draft": answer2_raw,
                    "answer_validation": {"ok": False, "reason": reason2},
                    "retrieval_count": ev_debug.get("num_results", 0),
                },
            )
        answer = answer2
        reason = "ok_after_repair"

    evidence_md = _render_evidence_section(evidence)
    final = "\n\n".join([answer, evidence_md]).strip()

    return DocQARunResult(
        answer=final,
        decision="answer",
        refusal_reason=None,
        evidence=evidence,
        features=features,
        p_supported=gate.p_supported,
        threshold=config.threshold,
        claims=[],
        claim_verifications=[],
        latency_s=time.time() - started,
        debug={
            "evidence": ev_debug,
            "gate": gate.__dict__,
            "draft": answer_raw,
            "answer_validation": {"ok": True, "reason": reason},
            "retrieval_count": ev_debug.get("num_results", 0),
            "llm_used": True,
        },
    )
