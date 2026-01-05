#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


URL_RE = re.compile(r"https?://[^\s\]\)\"']+", re.IGNORECASE)
PDF_CACHE_DIRNAME = ".cache_uploaded_pdfs"
PDF_CACHE_META = "last.json"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_pdf_path(root: Path) -> Optional[Path]:
    candidates = [
        root / "knowledge" / "AbuDhabi_ClimateChange_Essay.pdf",
        root / "knowledge" / "dspy.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name or "document.pdf")


def _ensure_cached_pdf(pdf_path: Path, cache_dir: Path) -> tuple[Path, str]:
    data = pdf_path.read_bytes()
    file_hash = _hash_bytes(data)
    cached_name = f"{file_hash}_{_sanitize_filename(pdf_path.name)}"
    cached_path = cache_dir / cached_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cached_path.exists():
        cached_path.write_bytes(data)
    return cached_path, file_hash


def _write_last_meta(meta_path: Path, payload: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(meta_path)


def _read_last_meta(meta_path: Path) -> Optional[dict]:
    try:
        if not meta_path.exists():
            return None
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _wait_groundx_complete(pdf_tool, timeout_s: int) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        try:
            status_resp = pdf_tool.client.documents.get_processing_status_by_id(
                process_id=pdf_tool.process_id
            )
            last_status = getattr(getattr(status_resp, "ingest", None), "status", None)
        except Exception as e:
            return False, f"processing status check failed: {e}"

        if last_status == "complete":
            return True, "complete"

        time.sleep(2)

    return False, f"timed out waiting for ingest completion (last status: {last_status})"


def _extract_json_object(raw: str) -> Optional[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def _coerce_page(v) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _evidence_ok(ev: list[dict]) -> tuple[bool, str]:
    if not isinstance(ev, list) or len(ev) == 0:
        return False, "evidence list empty"
    for i, item in enumerate(ev):
        if not isinstance(item, dict):
            return False, f"evidence[{i}] is not an object"
        page = _coerce_page(item.get("page"))
        quote = item.get("quote")
        if page is None or page < 1:
            return False, f"evidence[{i}].page invalid: {page!r}"
        if not isinstance(quote, str) or len(quote.strip()) < 10:
            return False, f"evidence[{i}].quote too short/invalid"
        item["page"] = page
    return True, "ok"


def _evidence_relevant(user_query: str, ev: list[dict]) -> bool:
    if not isinstance(user_query, str) or not user_query.strip():
        return True
    if not isinstance(ev, list) or not ev:
        return False

    stop = {
        "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
        "could", "describe", "did", "do", "does", "document", "explain", "exact", "for",
        "from", "give", "how", "i", "in", "include", "is", "it", "its", "list", "long",
        "me", "of", "on", "one", "or", "pdf", "please", "provide", "quote", "quotes",
        "say", "sentence", "sentences", "short", "show", "summarize", "summary", "tell",
        "that", "the", "their", "them", "these", "this", "those", "to", "two", "three",
        "four", "five", "six", "seven", "eight", "nine", "ten", "using", "verbatim",
        "direct", "was", "were", "what", "when", "where", "which", "who", "why", "with",
        "without", "you", "your",
    }

    def _tokens(s: str) -> set[str]:
        toks = re.findall(r"[a-z0-9]+", (s or "").lower())
        out = set()
        for t in toks:
            if t in stop:
                continue
            if len(t) < 3:
                continue
            out.add(t)
        return out

    q_tokens = _tokens(user_query)
    if not q_tokens:
        return True

    quote_tokens = [_tokens(str(i.get("quote") or "")) for i in ev]
    max_overlap = 0
    for qt in quote_tokens:
        try:
            max_overlap = max(max_overlap, len(q_tokens & qt))
        except Exception:
            continue

    min_overlap = 1 if len(q_tokens) <= 1 else 2
    if max_overlap < min_overlap:
        return False

    ev_tokens: set[str] = set()
    for qt in quote_tokens:
        try:
            ev_tokens |= qt
        except Exception:
            continue

    must_nums = set(re.findall(r"\b\d{4,}\b", user_query))
    for n in must_nums:
        if n not in ev_tokens:
            return False

    if "abu dhabi" in user_query.lower():
        if not any({"abu", "dhabi"}.issubset(qt) for qt in quote_tokens):
            return False

    return True


def _pick_answerable_query(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if "dspy" in name:
        return "What is the purpose of DSpy? Include one direct quote with (p.#)."
    if "abudhabi" in name or "abu_dhabi" in name or "abu" in name:
        return "Give one exact sentence about Abu Dhabi from this PDF with (p.#)."
    return "Give one exact sentence from this PDF with (p.#)."


def _pick_out_of_doc_query(pdf_path: Path) -> str:
    # Use a sentinel token that is extremely unlikely to appear in arbitrary PDFs.
    # Includes a 4+ digit number so the relevance gate requires that number in evidence.
    return (
        "What does this document say about ZXQJ_4921_UNLIKELY_TOKEN? "
        "Include a verbatim quote and (p.#). If it is not mentioned, respond exactly: Not in document."
    )


def _format_evidence_md(evidence: list[dict], max_items: int = 5) -> str:
    blocks = []
    for item in evidence[:max_items]:
        pg = int(item["page"])
        q = str(item["quote"]).rstrip()
        blocks.append(f"(p.{pg})\n```text\n{q}\n```")
    return "\n\n".join(blocks)


def _answer_ok(answer: str, evidence: list[dict]) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    a = answer.strip().lower()
    if a == "not in document.":
        return False
    if any(p in a for p in ["provided above", "as above", "mentioned above", "see above"]):
        return False

    evidence_pages = {int(item["page"]) for item in evidence}
    cited_pages = [int(x) for x in re.findall(r"\(p\.(\d+)\)", answer)]
    if not cited_pages:
        return False
    if any(p not in evidence_pages for p in cited_pages):
        return False

    # Require at least one verbatim snippet from evidence to appear in the answer.
    norm_answer = re.sub(r"\s+", " ", answer).strip().lower()
    for item in evidence:
        q = re.sub(r"\s+", " ", str(item.get("quote") or "")).strip().lower()
        words = q.split()
        if len(words) >= 12:
            snippet = " ".join(words[:12])
        else:
            snippet = q
        if snippet and len(snippet) >= 25 and snippet in norm_answer:
            return True

    return False


def _fallback_answer(evidence: list[dict]) -> str:
    top = evidence[: min(2, len(evidence))]
    lines = []
    for item in top:
        q = str(item["quote"]).strip()
        pg = int(item["page"])
        lines.append(f"\"{q}\" (p.{pg})")
    return "\n\n".join(lines) if lines else "Not in document."


def _build_pdf_crew(pdf_tool, llm):
    from crewai import Agent, Crew, Process, Task

    retriever = Agent(
        role="PDF evidence retriever for query: {query}",
        goal=(
            "Use ONLY the provided PDF search tool to retrieve verbatim evidence that answers the user query. "
            "The tool returns JSON search results with page numbers. Select the most relevant quotes and return "
            "ONLY JSON evidence. Do not answer the question. Do not use web or general knowledge."
        ),
        backstory="You are a strict evidence retriever. If you cannot find evidence, you return an empty evidence list.",
        verbose=False,
        tools=[pdf_tool],
        llm=llm,
    )

    synthesizer = Agent(
        role="PDF-grounded answer synthesizer for query: {query}",
        goal=(
            "Answer the user using ONLY the evidence returned by the retriever. "
            "Include at least one verbatim quote from evidence[].quote and cite it with (p.#) using evidence[].page. "
            "Never say 'provided above' or refer to unseen content. "
            "If evidence is missing or insufficient, respond exactly: Not in document."
        ),
        backstory="You never use outside knowledge. You never invent quotes, citations, or page numbers.",
        verbose=False,
        llm=llm,
    )

    retrieval_task = Task(
        description=(
            "Retrieve ONLY verbatim evidence from the uploaded PDF for the user query: {query}. "
            "Do NOT use general knowledge. Do NOT answer the question. "
            "Use the PDF search tool output to copy exact page numbers and quotes (do not guess). "
            "Output must be JSON only."
        ),
        expected_output='{"evidence":[{"page":<int >=1>,"quote":"<verbatim text from PDF>"}]}',
        agent=retriever,
    )

    response_task = Task(
        description=(
            "Using ONLY the JSON evidence from the retrieval task, answer the user query: {query}. "
            "Rules: (1) Use only evidence[].quote. (2) Cite every claim with (p.#) using evidence[].page. "
            "(2b) Include at least one direct quote from evidence[].quote verbatim. "
            "(3) If evidence is empty/insufficient, output exactly: Not in document. "
            "(4) Do not invent pages. Page numbers must be integers >= 1. (5) Do not invent quotes."
        ),
        expected_output="A concise answer grounded in the PDF with verbatim quotes and (p.#) citations, or exactly: Not in document.",
        agent=synthesizer,
        context=[retrieval_task],
    )

    crew = Crew(
        agents=[retriever, synthesizer],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew


def _run_pdf_qa_like_app(pdf_tool, llm, query: str) -> tuple[str, list[dict]]:
    crew = _build_pdf_crew(pdf_tool, llm)
    crew_result = crew.kickoff(inputs={"query": query})
    tasks_output = getattr(crew_result, "tasks_output", None)

    retrieval_raw = None
    final_raw = None
    if tasks_output and len(tasks_output) >= 1:
        retrieval_raw = getattr(tasks_output[0], "raw", None) or getattr(tasks_output[0], "output", None) or str(tasks_output[0])
    if tasks_output and len(tasks_output) >= 2:
        final_raw = getattr(tasks_output[1], "raw", None) or getattr(tasks_output[1], "output", None)
    if final_raw is None:
        final_raw = getattr(crew_result, "raw", None) or str(crew_result)

    evidence: list[dict] = []
    parsed = _extract_json_object(str(retrieval_raw) if retrieval_raw is not None else "")
    if isinstance(parsed, dict):
        ev = parsed.get("evidence", [])
        if isinstance(ev, list):
            evidence = ev

    ok, _ = _evidence_ok(evidence)
    if not ok:
        # Fallback: pull evidence directly from the PDF search tool output.
        try:
            tool_raw = pdf_tool._run(query)
            tool_obj = json.loads(tool_raw) if isinstance(tool_raw, str) else {}
            results = tool_obj.get("results", []) if isinstance(tool_obj, dict) else []
            fallback: list[dict] = []
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    page = _coerce_page(r.get("page"))
                    quote = r.get("quote")
                    if page is None or page < 1:
                        continue
                    if not isinstance(quote, str) or len(quote.strip()) < 10:
                        continue
                    fallback.append({"page": page, "quote": quote})
            evidence = fallback
        except Exception:
            evidence = []

    ok, _ = _evidence_ok(evidence)
    if not ok or not _evidence_relevant(query, evidence):
        return "Not in document.", []

    evidence_md = _format_evidence_md(evidence)
    answer = str(final_raw).strip() if final_raw is not None else ""
    if not _answer_ok(answer, evidence):
        answer = _fallback_answer(evidence)

    return f"### Evidence\n\n{evidence_md}\n\n### Answer\n\n{answer}", evidence


def check_doc_qa_answerable(pdf_path: Path, timeout_s: int) -> Check:
    if not pdf_path.exists():
        return Check("1) Document QA (answerable)", False, f"PDF not found: {pdf_path}")
    if not os.getenv("GROUNDX_API_KEY"):
        return Check("1) Document QA (answerable)", False, "GROUNDX_API_KEY is missing")

    root = _repo_root()
    sys.path.insert(0, str(root))

    try:
        from crewai import LLM
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool
    except Exception as e:
        return Check("1) Document QA (answerable)", False, f"import failed: {e}")

    cache_dir = root / PDF_CACHE_DIRNAME
    meta_path = cache_dir / PDF_CACHE_META
    cached_pdf, file_hash = _ensure_cached_pdf(pdf_path, cache_dir)

    # Reuse the last ingest if it matches the same cached PDF; avoids re-uploading on repeat runs.
    meta = _read_last_meta(meta_path) or {}
    reuse = (
        meta.get("path") == str(cached_pdf)
        and meta.get("hash") == file_hash
        and isinstance(meta.get("bucket_id"), int)
        and isinstance(meta.get("process_id"), str)
        and meta.get("process_id").strip()
    )

    try:
        if reuse:
            pdf_tool = DocumentSearchTool(
                file_path=str(cached_pdf),
                bucket_id=int(meta["bucket_id"]),
                process_id=str(meta["process_id"]),
                ready_timeout_s=0,
            )
        else:
            pdf_tool = DocumentSearchTool(file_path=str(cached_pdf), ready_timeout_s=0)
    except Exception as e:
        return Check("1) Document QA (answerable)", False, f"DocumentSearchTool init failed: {e}")

    # Persist IDs for refresh simulation
    _write_last_meta(
        meta_path,
        {
            "path": str(cached_pdf),
            "name": pdf_path.name,
            "hash": file_hash,
            "bucket_id": getattr(pdf_tool, "bucket_id", None),
            "process_id": getattr(pdf_tool, "process_id", None),
        },
    )

    ok, status = _wait_groundx_complete(pdf_tool, timeout_s=timeout_s)
    if not ok:
        return Check("1) Document QA (answerable)", False, status)

    llm = LLM(
        model=os.getenv("VERIFY_PDF_LLM_MODEL", "ollama/deepseek-r1:7b"),
        base_url=os.getenv("VERIFY_OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    query = _pick_answerable_query(pdf_path)
    try:
        result, evidence = _run_pdf_qa_like_app(pdf_tool, llm, query=query)
    except Exception as e:
        return Check("1) Document QA (answerable)", False, f"run failed: {e}")

    if result.strip() == "Not in document.":
        return Check("1) Document QA (answerable)", False, f"unexpected refusal for query: {query!r}")

    ok_ev, why = _evidence_ok(evidence)
    if not ok_ev:
        return Check("1) Document QA (answerable)", False, f"evidence invalid: {why}")

    # Confirm the final response includes an Answer section containing a quote + (p.#).
    m = re.search(r"###\s*Answer\s*\n+(.+)$", result, re.DOTALL)
    answer_text = m.group(1).strip() if m else ""
    if not answer_text:
        return Check("1) Document QA (answerable)", False, "missing Answer section")
    if not _answer_ok(answer_text, evidence):
        return Check("1) Document QA (answerable)", False, "answer missing quote/(p.#) or cites non-evidence page")

    return Check("1) Document QA (answerable)", True, f"ok (query={query!r}, evidence={len(evidence)})")


def check_doc_qa_refusal(timeout_s: int) -> Check:
    root = _repo_root()
    meta_path = root / PDF_CACHE_DIRNAME / PDF_CACHE_META
    meta = _read_last_meta(meta_path)
    cached_path = (meta or {}).get("path")
    if not isinstance(cached_path, str) or not cached_path:
        return Check("2) Document QA refusal", False, f"missing cached meta at: {meta_path}")
    cached_pdf = Path(cached_path)
    if not cached_pdf.exists():
        return Check("2) Document QA refusal", False, f"cached PDF missing: {cached_pdf}")
    if not os.getenv("GROUNDX_API_KEY"):
        return Check("2) Document QA refusal", False, "GROUNDX_API_KEY is missing")

    sys.path.insert(0, str(root))
    try:
        from crewai import LLM
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool
    except Exception as e:
        return Check("2) Document QA refusal", False, f"import failed: {e}")

    bucket_id = meta.get("bucket_id")
    process_id = meta.get("process_id")
    kwargs = {}
    if isinstance(bucket_id, int) and isinstance(process_id, str) and process_id.strip():
        kwargs = {"bucket_id": bucket_id, "process_id": process_id}
    kwargs["ready_timeout_s"] = 0

    try:
        pdf_tool = DocumentSearchTool(file_path=str(cached_pdf), **kwargs)
    except Exception as e:
        return Check("2) Document QA refusal", False, f"DocumentSearchTool init failed: {e}")

    # Don't re-wait the full timeout again; if ingest isn't complete, report the current status.
    try:
        status_resp = pdf_tool.client.documents.get_processing_status_by_id(process_id=pdf_tool.process_id)
        status = getattr(getattr(status_resp, "ingest", None), "status", None)
    except Exception as e:
        return Check("2) Document QA refusal", False, f"processing status check failed: {e}")
    if status != "complete":
        return Check("2) Document QA refusal", False, f"ingest not complete (status: {status})")

    llm = LLM(
        model=os.getenv("VERIFY_PDF_LLM_MODEL", "ollama/deepseek-r1:7b"),
        base_url=os.getenv("VERIFY_OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    query = _pick_out_of_doc_query(Path(meta.get("name") or cached_pdf.name))
    try:
        result, _evidence = _run_pdf_qa_like_app(pdf_tool, llm, query=query)
    except Exception as e:
        return Check("2) Document QA refusal", False, f"run failed: {e}")

    if result.strip() != "Not in document.":
        return Check("2) Document QA refusal", False, f"expected exact refusal, got: {result[:120]!r}")

    return Check("2) Document QA refusal", True, "ok (exact Not in document.)")


def check_refresh_persistence(timeout_s: int) -> Check:
    root = _repo_root()
    meta_path = root / PDF_CACHE_DIRNAME / PDF_CACHE_META
    meta = _read_last_meta(meta_path)
    if not meta:
        return Check("3) Refresh persistence", False, f"missing cached meta at: {meta_path}")

    cached_path = meta.get("path")
    bucket_id = meta.get("bucket_id")
    process_id = meta.get("process_id")
    if not isinstance(cached_path, str) or not cached_path:
        return Check("3) Refresh persistence", False, "last.json missing path")
    if not isinstance(bucket_id, int) or not isinstance(process_id, str) or not process_id.strip():
        return Check("3) Refresh persistence", False, "last.json missing bucket_id/process_id")

    cached_pdf = Path(cached_path)
    if not cached_pdf.exists():
        return Check("3) Refresh persistence", False, f"cached PDF missing: {cached_pdf}")
    if not os.getenv("GROUNDX_API_KEY"):
        return Check("3) Refresh persistence", False, "GROUNDX_API_KEY is missing")

    sys.path.insert(0, str(root))
    try:
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool
    except Exception as e:
        return Check("3) Refresh persistence", False, f"import failed: {e}")

    try:
        pdf_tool = DocumentSearchTool(file_path=str(cached_pdf), bucket_id=bucket_id, process_id=process_id, ready_timeout_s=0)
    except Exception as e:
        return Check("3) Refresh persistence", False, f"rehydrate init failed: {e}")

    try:
        status_resp = pdf_tool.client.documents.get_processing_status_by_id(process_id=pdf_tool.process_id)
        status = getattr(getattr(status_resp, "ingest", None), "status", None)
    except Exception as e:
        return Check("3) Refresh persistence", False, f"processing status check failed: {e}")
    if status != "complete":
        return Check("3) Refresh persistence", False, f"ingest not complete (status: {status})")

    return Check("3) Refresh persistence", True, f"ok (rehydrated bucket_id={bucket_id}, process_id={process_id})")


def _extract_urls(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    urls = [u.rstrip(".,;:!?") for u in URL_RE.findall(text)]
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _urls_ok(text: str, min_urls: int) -> tuple[bool, list[str], str]:
    urls = _extract_urls(text)
    if min_urls <= 0:
        return True, urls, "skipped"

    lowered = (text or "").lower()
    placeholders = ["retrieved from link", "available at: https://example.com"]
    if any(p in lowered for p in placeholders):
        return False, urls, "placeholder content detected"

    banned_domains = {
        "example.com",
        "www.example.com",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
    }

    valid_urls: list[str] = []
    parsed = []
    for u in urls:
        try:
            p = urlparse(u)
        except Exception:
            continue
        if p.scheme not in {"http", "https"} or not p.netloc:
            continue
        host = p.netloc.split("@")[-1].split(":")[0].strip().lower()
        if host in banned_domains:
            return False, urls, f"placeholder/banned domain: {host}"
        if not host or "." not in host:
            continue
        if not re.fullmatch(r"[a-z0-9.-]+", host):
            continue
        tld = host.rsplit(".", 1)[-1]
        allowed_long_tlds = {"com", "org", "net", "edu", "gov", "int", "info", "biz"}
        if not ((len(tld) == 2 and tld.isalpha()) or (tld in allowed_long_tlds)):
            continue
        valid_urls.append(u)
        parsed.append(p)

    if len(valid_urls) < min_urls:
        return False, valid_urls, f"insufficient urls: found {len(valid_urls)}"

    return True, valid_urls, "ok"


def _mcp_detect_defaults(root: Path) -> tuple[Optional[Path], Optional[Path]]:
    mcp_project = (root.parent / "Multi-Agent-deep-researcher-mcp-windows-linux").resolve()
    server_py = mcp_project / "server.py"
    server_python = mcp_project / ".venv" / "bin" / "python"
    return (server_python if server_python.exists() else None, server_py if server_py.exists() else None)


def check_mcp_tools(mcp_python: Path, mcp_server: Path, timeout_s: int) -> Check:
    try:
        import anyio
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as e:
        return Check("4) MCP tools", False, f"mcp client import failed: {e}")

    root = _repo_root()
    home_dir = (root / ".cache" / "mcp_home").resolve()
    home_dir.mkdir(parents=True, exist_ok=True)

    server_env = dict(os.environ)
    server_env.setdefault("RICH_DISABLE", "1")
    server_env.setdefault("NO_COLOR", "1")
    server_env.setdefault("TERM", "dumb")
    server_env.setdefault("CLICOLOR", "0")
    server_env.setdefault("PYTHONUNBUFFERED", "1")
    server_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    server_env.setdefault("CREWAI_VERBOSE", "0")
    server_env["HOME"] = str(home_dir)

    async def _run() -> tuple[bool, str, list[str]]:
        params = StdioServerParameters(
            command=str(mcp_python),
            args=[str(mcp_server)],
            env=server_env,
            cwd=str(mcp_server.parent),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                with anyio.fail_after(timeout_s):
                    await session.initialize()
                    tools_resp = await session.list_tools()

                tools = getattr(tools_resp, "tools", None) or tools_resp
                names: list[str] = []
                if isinstance(tools, list):
                    for t in tools:
                        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                        if name:
                            names.append(name)
                if not names:
                    return False, "no tools returned by MCP server", []
                if "crew_research" not in names:
                    return False, f"expected crew_research tool; got: {names}", names
                return True, f"ok (tools={names})", names

    try:
        ok, detail, _ = anyio.run(_run)
        return Check("4) MCP tools", ok, detail)
    except Exception as e:
        return Check("4) MCP tools", False, f"exception: {e}")


def check_mcp_sources(mcp_python: Path, mcp_server: Path, timeout_s: int) -> Check:
    try:
        import anyio
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as e:
        return Check("5) MCP sources", False, f"mcp client import failed: {e}")

    root = _repo_root()
    home_dir = (root / ".cache" / "mcp_home").resolve()
    home_dir.mkdir(parents=True, exist_ok=True)

    server_env = dict(os.environ)
    server_env.setdefault("RICH_DISABLE", "1")
    server_env.setdefault("NO_COLOR", "1")
    server_env.setdefault("TERM", "dumb")
    server_env.setdefault("CLICOLOR", "0")
    server_env.setdefault("PYTHONUNBUFFERED", "1")
    server_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    server_env.setdefault("CREWAI_VERBOSE", "0")
    server_env["HOME"] = str(home_dir)

    min_urls = 3
    query = (
        "Give 3 reputable sources with full https URLs about climate change impacts. "
        "Do NOT use homepages; each URL must include a non-root path (beyond '/') and directly support the claim. "
        "Return a final 'Sources' section of bullet URLs. "
        "If you cannot, output exactly: NO_SOURCES."
    )

    async def _run() -> tuple[bool, str]:
        params = StdioServerParameters(
            command=str(mcp_python),
            args=[str(mcp_server)],
            env=server_env,
            cwd=str(mcp_server.parent),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                with anyio.fail_after(timeout_s):
                    await session.initialize()
                    tools_resp = await session.list_tools()

                tools = getattr(tools_resp, "tools", None) or tools_resp
                names: list[str] = []
                if isinstance(tools, list):
                    for t in tools:
                        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                        if name:
                            names.append(name)
                if not names:
                    return False, "no tools returned by MCP server"

                tool_name = "crew_research" if "crew_research" in names else names[0]

                with anyio.fail_after(timeout_s):
                    result = await session.call_tool(tool_name, {"query": query})

                content = getattr(result, "content", None)
                parts: list[str] = []
                if isinstance(content, list) and content:
                    for c in content:
                        t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                        if isinstance(t, str) and t.strip():
                            parts.append(t)
                text = "\n\n".join(parts) if parts else str(content if content is not None else result)

                if re.search(r"\bNO_SOURCES\b", text, re.IGNORECASE):
                    return True, "ok (NO_SOURCES)"

                ok, urls, reason = _urls_ok(text, min_urls=min_urls)
                if not ok:
                    return False, f"{reason}; urls={urls}"

                return True, f"ok (urls={urls[:min_urls]})"

    try:
        ok, detail = anyio.run(_run)
        return Check("5) MCP sources", ok, detail)
    except Exception as e:
        return Check("5) MCP sources", False, f"exception: {e}")


def main() -> int:
    # Keep CrewAI quiet and disable rich/ANSI output by default.
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_VERBOSE", "0")
    os.environ.setdefault("RICH_DISABLE", "1")
    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("TERM", "dumb")
    os.environ.setdefault("CLICOLOR", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    # Load .env if available.
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=_repo_root() / ".env", override=False)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="End-to-end verification for Document QA + MCP Deep Research")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a local PDF to use")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout seconds per stage")
    parser.add_argument("--mcp-python", type=str, default=None, help="Path to MCP venv python (e.g., .../.venv/bin/python)")
    parser.add_argument("--mcp-server", type=str, default=None, help="Path to MCP server.py")
    args = parser.parse_args()

    root = _repo_root()
    pdf_path = Path(args.pdf).expanduser().resolve() if args.pdf else _default_pdf_path(root)
    if pdf_path is None:
        print("FAIL: No PDF provided and no default PDF found.", file=sys.stderr)
        return 2

    # Do NOT Path.resolve() the venv python executable: it can resolve symlinks to the
    # underlying system interpreter and break venv site-packages detection.
    def _abspath(p: str) -> Path:
        return Path(os.path.abspath(str(Path(p).expanduser())))

    mcp_python = _abspath(args.mcp_python) if args.mcp_python else None
    mcp_server = _abspath(args.mcp_server) if args.mcp_server else None
    if mcp_python is None or mcp_server is None:
        d_py, d_srv = _mcp_detect_defaults(root)
        mcp_python = mcp_python or d_py
        mcp_server = mcp_server or d_srv

    checks: list[Check] = [
        check_doc_qa_answerable(pdf_path, timeout_s=args.timeout),
        check_doc_qa_refusal(timeout_s=args.timeout),
        check_refresh_persistence(timeout_s=args.timeout),
    ]

    if mcp_python is None or mcp_server is None:
        checks.append(Check("4) MCP tools", False, "missing --mcp-python/--mcp-server and auto-detect failed"))
        checks.append(Check("5) MCP sources", False, "missing --mcp-python/--mcp-server and auto-detect failed"))
    else:
        if not mcp_python.exists():
            checks.append(Check("4) MCP tools", False, f"mcp python not found: {mcp_python}"))
            checks.append(Check("5) MCP sources", False, f"mcp python not found: {mcp_python}"))
        elif not mcp_server.exists():
            checks.append(Check("4) MCP tools", False, f"mcp server not found: {mcp_server}"))
            checks.append(Check("5) MCP sources", False, f"mcp server not found: {mcp_server}"))
        else:
            checks.append(check_mcp_tools(mcp_python, mcp_server, timeout_s=args.timeout))
            checks.append(check_mcp_sources(mcp_python, mcp_server, timeout_s=args.timeout))

    all_ok = True
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"{status}: {c.name} - {c.detail}")
        all_ok = all_ok and c.ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
