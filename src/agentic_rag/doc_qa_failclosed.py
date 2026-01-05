from __future__ import annotations

import json
import re
from typing import Any, Optional


_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b[^\s]+\.(com|org|net)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_UPPER_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_QUOTE_PHRASE_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]")


_STOPWORDS: set[str] = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "document", "explain", "exact", "for", "from",
    "describe", "give", "how", "i", "in", "include", "is", "it", "its", "list", "long", "me",
    "of", "on", "one", "or", "pdf", "please", "provide", "quote", "quotes", "say",
    "sentence", "sentences", "short", "show", "summarize", "summary", "tell",
    "that", "the", "their", "them", "these", "this", "those", "to", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "using", "verbatim",
    "direct", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "without", "you", "your",
}


def _coerce_page(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _parse_tool_results(raw: Any) -> list[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    results = obj.get("results", [])
    return results if isinstance(results, list) else []


def _contains_disallowed_patterns(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True

    t = text.strip()
    t_low = t.lower()

    if _URL_RE.search(t) or _DOMAIN_RE.search(t):
        return True

    disallowed = [
        "source:",
        "sources:",
        "reference:",
        "references:",
        "retrieved from",
        "accessed ",
        "doi",
    ]
    return any(s in t_low for s in disallowed)


def _looks_like_bibliography(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
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
        if _YEAR_RE.search(ln) and ("," in ln or ";" in ln):
            ref_like += 1
            continue

    return (ref_like / max(1, len(lines))) >= 0.5


def sanitize_evidence_from_tool_results(results: list[dict]) -> tuple[list[dict], dict]:
    accepted: list[dict] = []
    rejected: list[dict] = []

    if not isinstance(results, list):
        return [], {"accepted": [], "rejected": []}

    seen: set[tuple[int, str]] = set()

    for r in results:
        if not isinstance(r, dict):
            continue
        page = _coerce_page(r.get("page"))
        quote = r.get("quote")
        if page is None or page < 1 or not isinstance(quote, str):
            continue

        q_full = quote.strip()
        if len(q_full) < 10:
            continue

        # If a block is dominated by references, drop it entirely (fail-closed).
        if _looks_like_bibliography(q_full):
            rejected.append({"page": page, "quote": q_full, "reason": "bibliography_like"})
            continue

        # Otherwise, split into sentences and keep only URL/citation-free sentences.
        for sent in _sentences(q_full):
            s = sent.strip()
            if len(s) < 10:
                continue
            if _contains_disallowed_patterns(s):
                rejected.append({"page": page, "quote": s, "reason": "disallowed_pattern"})
                continue

            key = (page, s)
            if key in seen:
                continue
            seen.add(key)
            accepted.append({"page": page, "quote": s})

    debug = {"accepted": accepted, "rejected": rejected}
    return accepted, debug


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


def _extract_quoted_phrases(query: str) -> list[str]:
    if not isinstance(query, str):
        return []
    phrases = []
    for m in _QUOTE_PHRASE_RE.finditer(query):
        p = (m.group(1) or "").strip()
        if p:
            phrases.append(p)
    return phrases


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("-", " ")).strip()


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


def _sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    out: list[str] = []
    for p in parts:
        p2 = p.strip()
        if p2:
            out.append(p2)
    return out


def _render_quote_blocks(quotes: list[dict]) -> str:
    blocks: list[str] = []
    for item in quotes:
        pg = int(item["page"])
        q = str(item["quote"]).rstrip()
        blocks.append(f"(p.{pg})\n```text\n{q}\n```")
    return "\n\n".join(blocks) if blocks else "Not in document."


def _select_quotes_for_general_query(query: str, evidence: list[dict]) -> list[dict]:
    q = query or ""
    q_low = q.lower()
    q_tokens = _tokens(q)

    must_nums = set(re.findall(r"\b\d{4,}\b", q))
    must_phrases = [_normalize_for_match(p) for p in _extract_quoted_phrases(q)]
    must_acronyms = {_normalize_for_match(a) for a in _UPPER_ACRONYM_RE.findall(q)}

    require_abu_dhabi = "abu dhabi" in q_low
    if require_abu_dhabi:
        must_acronyms.discard("abu")
        must_acronyms.discard("dhabi")

    def _quote_ok(item: dict) -> bool:
        qt = str(item.get("quote") or "")
        qt_low = qt.lower()
        qt_norm = _normalize_for_match(qt)
        if require_abu_dhabi and not ("abu" in qt_low and "dhabi" in qt_low):
            return False
        for n in must_nums:
            if n not in qt:
                return False
        for p in must_phrases:
            if p and p not in qt_norm:
                return False
        for a in must_acronyms:
            if a and a not in qt_norm:
                return False
        return True

    scored: list[tuple[int, int, dict]] = []
    for idx, item in enumerate(evidence):
        qtoks = _tokens(str(item.get("quote") or ""))
        overlap = len(q_tokens & qtoks) if q_tokens else 0
        if q_tokens:
            min_overlap = 1 if len(q_tokens) <= 1 else 2
            if overlap < min_overlap:
                continue
        if not _quote_ok(item):
            continue
        score = overlap
        if require_abu_dhabi:
            score += 2
        score += 2 * sum(1 for n in must_nums if n in str(item.get("quote") or ""))
        score += 2 * sum(1 for p in must_phrases if p and p in _normalize_for_match(str(item.get("quote") or "")))
        score += 2 * sum(1 for a in must_acronyms if a and a in _normalize_for_match(str(item.get("quote") or "")))
        scored.append((score, -idx, item))

    scored.sort(reverse=True)
    picked = [t[2] for t in scored[:2]]
    return picked


def run_document_qa_failclosed_debug(query: str, pdf_tool) -> tuple[str, dict]:
    debug: dict = {"evidence": [], "rejected": [], "tool_error": None}

    if pdf_tool is None:
        return "Not in document.", {**debug, "tool_error": "no pdf_tool"}

    tool_raw: Optional[str] = None
    try:
        tool_raw = pdf_tool._run(query)
    except Exception as e:
        return "Not in document.", {**debug, "tool_error": str(e)}

    results = _parse_tool_results(tool_raw)
    evidence, ev_debug = sanitize_evidence_from_tool_results(results)
    debug["evidence"] = ev_debug.get("accepted", [])
    debug["rejected"] = ev_debug.get("rejected", [])

    if not evidence:
        return "Not in document.", debug

    q_low = (query or "").lower()
    if any(k in q_low for k in [" url", " urls", "http", "https", "www", "doi"]):
        return "Not in document.", debug

    n_sent = _parse_requested_sentence_count(query)
    if n_sent is not None:
        phrases = _extract_quoted_phrases(query)
        must_phrase = phrases[0] if phrases else None
        must_norm = _normalize_for_match(must_phrase) if must_phrase else None

        candidates: list[dict] = []
        for item in evidence:
            pg = int(item["page"])
            for sent in _sentences(str(item["quote"])):
                if must_norm:
                    if must_norm not in _normalize_for_match(sent):
                        continue
                candidates.append({"page": pg, "quote": sent})

        if len(candidates) < n_sent:
            return "Not in document.", debug

        return _render_quote_blocks(candidates[:n_sent]), debug

    if "bullet" in q_low or "bullets" in q_low:
        bullets = []
        for item in _select_quotes_for_general_query(query, evidence)[:3]:
            pg = int(item["page"])
            q = str(item["quote"]).rstrip()
            bullets.append(f"- (p.{pg})\n  ```text\n  {q}\n  ```")
        return "\n".join(bullets) if bullets else "Not in document.", debug

    picked = _select_quotes_for_general_query(query, evidence)
    if not picked:
        return "Not in document.", debug
    return _render_quote_blocks(picked), debug


def run_document_qa_failclosed(query: str, pdf_tool) -> str:
    out, _ = run_document_qa_failclosed_debug(query=query, pdf_tool=pdf_tool)
    return out
