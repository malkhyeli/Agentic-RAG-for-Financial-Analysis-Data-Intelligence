from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from dotenv import load_dotenv
from groundx import Document, GroundX
from groundx.core.api_error import ApiError
from pydantic import BaseModel, ConfigDict, Field

# pypdf fallback for when GroundX doesn't extract table content
try:
    from pypdf import PdfReader as _PdfReader
except ImportError:
    _PdfReader = None

try:
    _ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except Exception:
    pass

from src.utils.env import get_groundx_api_key


_LOCAL_STOPWORDS: set[str] = {
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

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_URLISH_RE = re.compile(r"https?://|www\.|\b[^\s]+\.(com|org|net)\b/\S+", re.IGNORECASE)

# Heuristic: sequences of at least two capitalized words, with an optional middle initial.
_NAME_LIKE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b")

_TITLE_KEYWORDS: set[str] = {
    "chief",
    "officer",
    "president",
    "vice",
    "ceo",
    "cfo",
    "chair",
    "director",
    "general",
    "counsel",
    "controller",
    "accounting",
    "principal",
    "senior",
}

_PEOPLE_KEYWORDS: set[str] = {
    "student",
    "students",
    "member",
    "members",
    "team",
    "name",
    "names",
    "who",
    "person",
    "people",
    "author",
    "authors",
    "participant",
    "participants",
    "executive",
    "officer",
    "officers",
    "director",
    "directors",
    "board",
    "management",
    "leadership",
    "ceo",
    "cfo",
    "chair",
}

_PEOPLE_SECTION_MARKERS: list[str] = [
    "executive officer",
    "executive officers",
    "officers and directors",
    "executive officers and directors",
    "information about our executive officers",
    "board of directors",
    "management",
    "leadership",
    "team member",
    "members",
    "student",
    "author",
    "participant",
    "name age position",
    "name",
    "signed",
    "prepared by",
    "submitted by",
]


def _normalize_score(v: Any) -> float:
    try:
        s = float(v)
    except Exception:
        return 0.0
    if not (s > 0.0):
        return 0.0
    # GroundX scores are typically 0..1, but some APIs return 0..100.
    if 1.0 < s <= 100.0:
        return s / 100.0
    if s > 100.0:
        return 1.0
    return min(1.0, s)


def _tokenize_terms(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    out: list[str] = []
    for t in toks:
        if len(t) < 3 or t in _LOCAL_STOPWORDS:
            continue
        out.append(t)
        if t.endswith("s") and len(t) > 3:
            out.append(t[:-1])
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def _is_people_query(query: str, terms: list[str]) -> bool:
    q_low = (query or "").lower()
    if "executive officer" in q_low or "board of directors" in q_low:
        return True
    return any(t in _PEOPLE_KEYWORDS for t in (terms or []))


def _count_name_like(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(_NAME_LIKE_RE.findall(text))


def _contains_urlish(text: str) -> bool:
    return bool(_URLISH_RE.search(text or ""))


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""

    query: str = Field(..., description="Query to search the document.")


class MissingApiKeyError(RuntimeError):
    """Raised when a required API key is missing or empty."""

    def __init__(self, message: str = "Missing required API key.") -> None:
        super().__init__(message)


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = (
        "Search the document for the given query and return JSON results with page numbers and quotes."
    )
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        file_path: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        pdf: Optional[str] = None,
        bucket_id: Optional[int] = None,
        process_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        ready_timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not file_path and pdf:
            file_path = pdf

        chosen_key = (api_key or get_groundx_api_key() or "").strip()
        if not chosen_key:
            raise MissingApiKeyError(
                "GROUNDX_API_KEY is not set. Add it to .env (recommended) or export it in your shell."
            )

        if not file_path and (bucket_id is None or process_id is None):
            raise ValueError(
                "DocumentSearchTool requires file_path (or pdf=...) unless resuming with bucket_id and process_id."
            )

        resolved_path = str(Path(file_path).expanduser().resolve()) if file_path else None
        self.file_path = resolved_path

        self.client = GroundX(api_key=chosen_key)
        self._has_lookup: bool = hasattr(self.client, "documents") and hasattr(
            getattr(self.client, "documents"), "lookup"
        )

        self.document_id: Optional[str] = None
        self.document_ids: list[str] = [str(d).strip() for d in (document_ids or []) if str(d).strip()]
        self.bucket_file_count: Optional[int] = None
        self.last_search_scope: Optional[str] = None
        self.last_search_error: Optional[str] = None
        self.last_poll_status: Optional[str] = None
        self.elapsed_wait_s: float = 0.0
        self.in_progress: bool = False
        self.ready: bool = False

        self.ready_timeout_s = float(ready_timeout_s) if ready_timeout_s is not None else float(
            os.getenv("GROUNDX_READY_TIMEOUT_S", 900)
        )
        self.poll_interval_s = float(poll_interval_s) if poll_interval_s is not None else float(
            os.getenv("GROUNDX_POLL_INTERVAL_S", 2.0)
        )
        self.max_retries = int(max_retries) if max_retries is not None else int(os.getenv("GROUNDX_RETRIES", 3))

        if bucket_id is not None and process_id is not None:
            self.bucket_id = int(bucket_id)
            self.process_id = str(process_id)
        else:
            if not self.file_path:
                raise ValueError(
                    "DocumentSearchTool requires file_path (or pdf=...) unless resuming with bucket_id and process_id."
                )
            if not Path(self.file_path).is_file():
                raise FileNotFoundError(f"PDF not found: {self.file_path}")
            self.bucket_id = self._create_bucket()
            self.process_id = self._upload_document()

        self._lookup_document_ids()
        self.ready = self._wait_until_ready(timeout_s=self.ready_timeout_s)
        self._lookup_document_ids()
        self.bucket_file_count = self._refresh_bucket_file_count()
        if self.document_ids and not self.document_id:
            self.document_id = self.document_ids[0]
        self.in_progress = not self.ready

    def _with_retries(self, func, *args, **kwargs):
        attempts = 0
        delay = float(self.poll_interval_s or 1.0)
        while True:
            try:
                return func(*args, **kwargs)
            except ApiError as e:
                status = getattr(e, "status_code", None)
                if status in (401, 403):
                    raise
                if status is not None and status < 500:
                    raise
                attempts += 1
                if attempts > self.max_retries:
                    raise
                time.sleep(delay + random.uniform(0, 0.2))
                delay *= 2
            except Exception:
                attempts += 1
                if attempts > self.max_retries:
                    raise
                time.sleep(delay + random.uniform(0, 0.2))
                delay *= 2

    def _create_bucket(self) -> int:
        suffix = f"{int(time.time())}"
        bucket_name = f"agentic_rag-{suffix}-{random.randint(0, 0xFFFF):04x}"
        response = self._with_retries(self.client.buckets.create, name=bucket_name)
        return int(response.bucket.bucket_id)

    def _upload_document(self) -> str:
        if not self.file_path:
            raise ValueError("file_path is required to upload a document")
        ingest = self._with_retries(
            self.client.ingest,
            documents=[
                Document(
                    bucket_id=self.bucket_id,
                    file_name=os.path.basename(self.file_path),
                    file_path=self.file_path,
                    file_type="pdf",
                    search_data=dict(key="value"),
                )
            ],
        )

        process_id = None
        for path in (("ingest", "process_id"), ("process_id",), ("ingest", "processId"), ("processId",)):
            try:
                cur = ingest
                for key in path:
                    cur = getattr(cur, key)
                process_id = cur
                if process_id:
                    break
            except Exception:
                continue
        if not process_id and isinstance(ingest, dict):
            process_id = ingest.get("process_id") or ingest.get("processId")
            if not process_id and isinstance(ingest.get("ingest"), dict):
                process_id = ingest["ingest"].get("process_id") or ingest["ingest"].get("processId")

        if not process_id:
            raise RuntimeError("GroundX ingest response missing process_id")
        return str(process_id)

    def _extract_document_ids(self, docs: Any) -> list[str]:
        ids: list[str] = []
        for doc in docs or []:
            candidate = None
            if isinstance(doc, dict):
                candidate = doc.get("document_id") or doc.get("documentId") or doc.get("id")
            else:
                for attr in ("document_id", "documentId", "id"):
                    candidate = getattr(doc, attr, None)
                    if candidate:
                        break
            if candidate is None:
                continue
            s = str(candidate).strip()
            if s:
                ids.append(s)
        return ids

    def _lookup_document_ids(self) -> list[str]:
        if not getattr(self, "_has_lookup", False):
            return list(getattr(self, "document_ids", []) or [])

        lookup_id: Any = (
            self.process_id if getattr(self, "process_id", None) is not None else getattr(self, "bucket_id", None)
        )
        if lookup_id is None:
            return list(getattr(self, "document_ids", []) or [])

        try:
            lookup = self._with_retries(self.client.documents.lookup, lookup_id)
        except Exception:
            bucket_id = getattr(self, "bucket_id", None)
            if lookup_id != bucket_id and bucket_id is not None:
                try:
                    lookup = self._with_retries(self.client.documents.lookup, bucket_id)
                except Exception:
                    return list(getattr(self, "document_ids", []) or [])
            else:
                return list(getattr(self, "document_ids", []) or [])

        docs = None
        try:
            docs = getattr(lookup, "documents", None)
            if docs is None:
                docs = getattr(getattr(lookup, "ingest", None), "documents", None)
        except Exception:
            docs = None

        if docs is None and isinstance(lookup, dict):
            docs = lookup.get("documents") or (lookup.get("ingest") or {}).get("documents")

        if docs and not isinstance(docs, (list, tuple)):
            docs = [docs]

        ids = self._extract_document_ids(docs or [])
        if not ids:
            simple_ids = None
            for attr in ("document_ids", "documentIds"):
                if isinstance(lookup, dict):
                    simple_ids = lookup.get(attr)
                else:
                    simple_ids = getattr(lookup, attr, None)
                if simple_ids:
                    break
            if isinstance(simple_ids, (list, tuple)):
                ids = [str(x).strip() for x in simple_ids if str(x).strip()]

        if ids:
            self.document_ids = ids
            self.document_id = self.document_id or ids[0]
        return list(getattr(self, "document_ids", []) or [])

    def _refresh_bucket_file_count(self) -> Optional[int]:
        if getattr(self, "bucket_file_count", None) is not None:
            return self.bucket_file_count
        if not getattr(self, "_has_lookup", False):
            return self.bucket_file_count
        bid = getattr(self, "bucket_id", None)
        if bid is None:
            return self.bucket_file_count
        try:
            lookup = self._with_retries(self.client.documents.lookup, bid, n=1)
            docs = getattr(lookup, "documents", None)
            if docs is None and isinstance(lookup, dict):
                docs = lookup.get("documents")
            if docs is None:
                return self.bucket_file_count
            if not isinstance(docs, (list, tuple)):
                docs = [docs]
            self.bucket_file_count = len(docs)
        except Exception:
            pass
        return self.bucket_file_count

    def _wait_until_ready(self, timeout_s: float = 120.0) -> bool:
        start = time.monotonic()
        in_progress_statuses = {"queued", "processing", "training", "running", "indexing"}

        def _poll_once() -> Optional[str]:
            try:
                status_response = self._with_retries(
                    self.client.documents.get_processing_status_by_id,
                    process_id=self.process_id,
                )
                status = getattr(getattr(status_response, "ingest", None), "status", None)
                if isinstance(status, str):
                    return status.strip().lower()
            except Exception as exc:
                self.last_poll_status = f"error: {exc}"
                return None
            return None

        if timeout_s <= 0:
            last = _poll_once()
            self.last_poll_status = last
            self._lookup_document_ids()
            return last == "complete"

        last_status: Optional[str] = None
        while True:
            elapsed = time.monotonic() - start
            self.elapsed_wait_s = elapsed
            last_status = _poll_once()
            if last_status:
                self.last_poll_status = last_status

            self._lookup_document_ids()
            docs_ready = bool(self.document_ids) or not self._has_lookup

            if last_status == "complete":
                return docs_ready

            if last_status and last_status not in in_progress_statuses:
                return False

            if elapsed >= timeout_s:
                return False

            sleep_s = float(self.poll_interval_s or 1.0) + random.uniform(0, 0.3)
            time.sleep(max(0.1, sleep_s))

    def _parse_search_results(self, search_response: Any) -> list[dict[str, Any]]:
        container = (
            getattr(search_response, "search", None)
            or getattr(search_response, "document_search", None)
            or getattr(search_response, "content_search", None)
            or search_response
        )
        raw_results = getattr(container, "results", None)
        if raw_results is None and isinstance(container, dict):
            raw_results = container.get("results")

        results: list[dict[str, Any]] = []
        for r in raw_results or []:
            text = None
            if isinstance(r, dict):
                val = r.get("text")
                if isinstance(val, str) and val.strip():
                    text = val.strip()
            else:
                val = getattr(r, "text", None)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
            if not text:
                continue

            page_num: Optional[int] = None

            pages = r.get("pages") if isinstance(r, dict) else getattr(r, "pages", None)
            if isinstance(pages, list) and pages:
                nums: list[int] = []
                for p in pages:
                    if isinstance(p, dict):
                        cand = p.get("number") or p.get("page_number") or p.get("pageNumber")
                    else:
                        cand = (
                            getattr(p, "number", None)
                            or getattr(p, "page_number", None)
                            or getattr(p, "pageNumber", None)
                        )
                    if isinstance(cand, int) and cand > 0:
                        nums.append(cand)
                    elif isinstance(cand, float) and cand.is_integer() and int(cand) > 0:
                        nums.append(int(cand))
                    elif isinstance(cand, str) and cand.strip().isdigit() and int(cand.strip()) > 0:
                        nums.append(int(cand.strip()))
                if nums:
                    page_num = min(nums)

            if page_num is None:
                bbs = None
                if isinstance(r, dict):
                    bbs = r.get("boundingBoxes") or r.get("bounding_boxes")
                else:
                    bbs = getattr(r, "bounding_boxes", None) or getattr(r, "boundingBoxes", None)
                if isinstance(bbs, list) and bbs:
                    nums: list[int] = []
                    for bb in bbs:
                        if isinstance(bb, dict):
                            cand = bb.get("page_number") or bb.get("pageNumber") or bb.get("page")
                        else:
                            cand = (
                                getattr(bb, "page_number", None)
                                or getattr(bb, "pageNumber", None)
                                or getattr(bb, "page", None)
                            )
                        if isinstance(cand, int) and cand > 0:
                            nums.append(cand)
                        elif isinstance(cand, float) and cand.is_integer() and int(cand) > 0:
                            nums.append(int(cand))
                        elif isinstance(cand, str) and cand.strip().isdigit() and int(cand.strip()) > 0:
                            nums.append(int(cand.strip()))
                    if nums:
                        counts: dict[int, int] = {}
                        for n in nums:
                            counts[n] = counts.get(n, 0) + 1
                        best = max(counts.values())
                        winners = [p for p, c in counts.items() if c == best]
                        page_num = min(winners) if winners else None

            if page_num is None:
                for attr in ("page", "page_number", "pageNumber"):
                    cand = r.get(attr) if isinstance(r, dict) else getattr(r, attr, None)
                    if isinstance(cand, int) and cand > 0:
                        page_num = cand
                        break
                    if isinstance(cand, float) and cand.is_integer() and int(cand) > 0:
                        page_num = int(cand)
                        break
                    if isinstance(cand, str) and cand.strip().isdigit() and int(cand.strip()) > 0:
                        page_num = int(cand.strip())
                        break

            score = None
            if isinstance(r, dict):
                score = r.get("score") if r.get("score") is not None else r.get("similarity")
            else:
                score = getattr(r, "score", None)
                if score is None:
                    score = getattr(r, "similarity", None)

            doc_id = None
            for attr in ("document_id", "documentId", "doc_id"):
                cand = r.get(attr) if isinstance(r, dict) else getattr(r, attr, None)
                if cand:
                    doc_id = cand
                    break

            results.append({"page": page_num, "quote": text, "score": score, "document_id": doc_id})

        return results

    def _search(self, query: str, n: int = 10) -> tuple[Any, list[dict[str, Any]]]:
        if not self.document_ids:
            self._lookup_document_ids()

        if self.document_ids and hasattr(self.client, "search") and hasattr(self.client.search, "documents"):
            try:
                # GroundX SDK expects document_ids as strings.
                ids_to_use = [str(d) for d in self.document_ids if str(d).strip()]
                raw = self._with_retries(
                    self.client.search.documents,
                    query=query,
                    document_ids=ids_to_use,
                    n=n,
                    verbosity=2,
                )
                self.last_search_scope = "documents"
                self.last_search_error = None
                return raw, self._parse_search_results(raw)
            except Exception as exc:
                self.last_search_scope = "documents_error"
                self.last_search_error = str(exc)

        raw = self._with_retries(
            self.client.search.content,
            id=self.bucket_id,
            query=query,
            n=n,
            verbosity=2,
        )
        self.last_search_scope = "bucket"
        self.last_search_error = None
        return raw, self._parse_search_results(raw)

    def _local_pdf_search(self, query: str, n: int = 10) -> list[dict[str, Any]]:
        """Fallback: search PDF locally using pypdf when GroundX doesn't find content.

        This handles cases where GroundX's parser doesn't extract table content,
        but the text is actually present in the PDF.
        """
        if _PdfReader is None:
            return []
        if not self.file_path or not Path(self.file_path).is_file():
            return []

        try:
            reader = _PdfReader(self.file_path)
        except Exception:
            return []

        query_terms = _tokenize_terms(query)
        term_set = set(query_terms)
        is_people_query = _is_people_query(query, query_terms)

        results: list[dict[str, Any]] = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue

            if not text.strip():
                continue

            text_lower = text.lower()

            page_tokens = set(_TOKEN_RE.findall(text_lower))
            overlap = len(term_set & page_tokens) if term_set else 0
            marker_hit = any(m in text_lower for m in _PEOPLE_SECTION_MARKERS) if is_people_query else False

            if overlap == 0 and not marker_hit:
                continue

            # Extract relevant lines containing query terms
            lines = text.split('\n')
            relevant_lines: list[str] = []

            for i, line in enumerate(lines):
                line_lower = line.lower()
                line_tokens = set(_TOKEN_RE.findall(line_lower))
                should_include = bool(term_set & line_tokens) if term_set else False
                if is_people_query and not should_include:
                    should_include = any(m in line_lower for m in _PEOPLE_SECTION_MARKERS)

                if should_include:
                    start = max(0, i - 1)
                    window_after = 2
                    if is_people_query:
                        window_after = 18
                        if any(
                            key in line_lower
                            for key in ("name age position", "information about", "executive officers", "board of directors")
                        ):
                            window_after = 35
                    end = min(len(lines), i + 1 + window_after)
                    context = '\n'.join(lines[start:end])
                    relevant_lines.append(context)

            if relevant_lines:
                # Deduplicate and combine
                seen = set()
                unique_quotes = []
                for quote in relevant_lines:
                    normalized = quote.strip()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        unique_quotes.append(normalized)

                def _score_quote(q: str) -> tuple[float, int, int, str]:
                    q_tokens = set(_TOKEN_RE.findall(q.lower()))
                    overlap_q = len(term_set & q_tokens) if term_set else 0
                    overlap_ratio = (overlap_q / max(1, len(term_set))) if term_set else 0.0

                    name_like = _count_name_like(q) if is_people_query else 0
                    title_hits = len(q_tokens & _TITLE_KEYWORDS) if is_people_query else 0

                    score = overlap_ratio
                    if is_people_query:
                        score += 0.25 * min(1.0, name_like / 4.0)
                        score += 0.10 * min(1.0, title_hits / 3.0)
                    if _contains_urlish(q):
                        score -= 0.15
                    score = max(0.0, min(1.0, score))
                    return (score, name_like, overlap_q, q)

                scored_quotes = sorted((_score_quote(q) for q in unique_quotes), reverse=True)
                for score, name_like, overlap_q, quote in scored_quotes[:5]:
                    results.append(
                        {
                            "page": page_num,
                            "quote": quote.strip(),
                            "score": score,
                            "document_id": None,
                            "source": "local_pypdf",
                            "name_like": name_like,
                            "overlap": overlap_q,
                        }
                    )

        # Sort by score (match ratio) descending
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[:n]

    def _run(self, query: str) -> str:
        query_terms = _tokenize_terms(query)
        term_set = set(query_terms)
        people_query = _is_people_query(query, query_terms)

        # GroundX may be in-progress when `ready_timeout_s=0`; still allow local search.
        status_complete = bool(getattr(self, "ready", False))
        if not status_complete:
            try:
                status_response = self._with_retries(
                    self.client.documents.get_processing_status_by_id,
                    process_id=self.process_id,
                )
                status = getattr(getattr(status_response, "ingest", None), "status", None)
                if isinstance(status, str):
                    self.last_poll_status = status
                    status_complete = status.strip().lower() == "complete"
            except Exception as exc:
                self.last_poll_status = f"error: {exc}"

        groundx_results: list[dict[str, Any]] = []
        if status_complete:
            try:
                _, groundx_results = self._search(query=query, n=25)
            except Exception:
                groundx_results = []

        local_results = self._local_pdf_search(query=query, n=25)

        local_has_names = any(_count_name_like(str(r.get("quote") or "")) >= 2 for r in local_results)
        groundx_has_names = any(_count_name_like(str(r.get("quote") or "")) >= 2 for r in groundx_results)
        primary = local_results if (people_query and local_has_names and not groundx_has_names) else groundx_results
        secondary = groundx_results if primary is local_results else local_results

        merged: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()

        def _add_items(items: list[dict[str, Any]]) -> None:
            for r in items or []:
                if not isinstance(r, dict):
                    continue
                quote = str(r.get("quote") or "").strip()
                if not quote:
                    continue
                page: Any = r.get("page")
                if isinstance(page, float) and page.is_integer():
                    page = int(page)
                if isinstance(page, str) and page.strip().isdigit():
                    page = int(page.strip())
                if not isinstance(page, int) or page < 1:
                    continue
                key = (page, quote)
                if key in seen:
                    continue
                seen.add(key)
                merged.append({**r, "page": page, "quote": quote})

        _add_items(primary)
        _add_items(secondary)

        def _rerank(r: dict[str, Any]) -> tuple[float, int, int]:
            quote = str(r.get("quote") or "")
            orig = _normalize_score(r.get("score"))
            q_tokens = set(_TOKEN_RE.findall(quote.lower()))
            overlap = len(q_tokens & term_set) if term_set else 0
            overlap_ratio = (overlap / max(1, len(term_set))) if term_set else 0.0
            name_like = _count_name_like(quote) if people_query else 0
            title_hits = len(q_tokens & _TITLE_KEYWORDS) if people_query else 0

            score = 0.55 * orig + 0.45 * overlap_ratio
            if people_query:
                score += 0.20 * min(1.0, name_like / 4.0)
                score += 0.10 * min(1.0, title_hits / 3.0)
            if _contains_urlish(quote):
                score -= 0.10
            score = max(0.0, min(1.0, score))
            r["score"] = score
            return (score, name_like, overlap)

        merged.sort(key=lambda r: _rerank(r), reverse=True)
        return json.dumps({"results": merged[:25]}, ensure_ascii=False)

    def debug_status(self) -> dict[str, Any]:
        try:
            status_response = self._with_retries(
                self.client.documents.get_processing_status_by_id,
                process_id=self.process_id,
            )
            ingest_status = getattr(getattr(status_response, "ingest", None), "status", None)
        except Exception as e:
            ingest_status = f"error: {e}"

        if not self.document_ids:
            self._lookup_document_ids()
        if self.bucket_file_count is None:
            self._refresh_bucket_file_count()

        return {
            "bucket_id": getattr(self, "bucket_id", None),
            "process_id": getattr(self, "process_id", None),
            "document_id": getattr(self, "document_id", None),
            "document_ids": getattr(self, "document_ids", []),
            "bucket_file_count": getattr(self, "bucket_file_count", None),
            "ingest_status": ingest_status,
            "last_search_scope": self.last_search_scope,
            "last_search_error": self.last_search_error,
            "last_poll_status": self.last_poll_status or ingest_status,
            "elapsed_wait_s": round(self.elapsed_wait_s, 2),
            "ready": bool(getattr(self, "ready", False)),
            "in_progress": bool(getattr(self, "in_progress", False)),
        }

    def _safe_serialize_raw(self, raw: Any) -> Any:
        if raw is None:
            return None
        for attr in ("model_dump", "dict", "to_dict"):
            fn = getattr(raw, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        try:
            return json.loads(json.dumps(raw, default=str, ensure_ascii=False))
        except Exception:
            return str(raw)

    def test_retrieval(self, query: str = "sea-level rise", n: int = 3) -> dict[str, Any]:
        out: dict[str, Any] = {"query": query, "status": self.debug_status(), "results": []}
        try:
            raw, results = self._search(query=query, n=n)
            out["results"] = results
            out["result_count"] = len(results)
            out["search_scope"] = self.last_search_scope
            out["raw"] = self._safe_serialize_raw(raw)
        except ApiError as e:
            out["error"] = {"status_code": getattr(e, "status_code", None), "body": getattr(e, "body", None)}
        except Exception as e:
            out["error"] = {"message": str(e)}
        return out
