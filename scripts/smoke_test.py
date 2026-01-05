#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


URL_RE = re.compile(r"https?://[^\s\]\)\"']+", re.IGNORECASE)


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


def _pick_pdf_query(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if "dspy" in name:
        return "What is the purpose of DSpy?"
    if "abudhabi" in name or "abu_dhabi" in name or "abu" in name:
        return "What does this document say about Abu Dhabi? Include one short quote."
    return "What is this document about? Include one short quote."


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


def _parse_evidence_json(raw: str) -> list[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return []

    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()

    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(s[start : end + 1])
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return []
    ev = obj.get("evidence", [])
    return ev if isinstance(ev, list) else []


def _evidence_ok(ev: list[dict]) -> tuple[bool, str]:
    if not isinstance(ev, list) or len(ev) == 0:
        return False, "evidence list empty"
    for i, item in enumerate(ev):
        if not isinstance(item, dict):
            return False, f"evidence[{i}] is not an object"
        page = item.get("page")
        quote = item.get("quote")
        if not isinstance(page, int) or page < 1:
            return False, f"evidence[{i}].page invalid: {page!r}"
        if not isinstance(quote, str) or len(quote.strip()) < 10:
            return False, f"evidence[{i}].quote too short/invalid"
    return True, "ok"


def check_pdf_qa(pdf_path: Path, timeout_s: int) -> Check:
    if not pdf_path.exists():
        return Check("PDF QA", False, f"PDF not found: {pdf_path}")

    if not os.getenv("GROUNDX_API_KEY"):
        return Check("PDF QA", False, "GROUNDX_API_KEY is missing")

    root = _repo_root()
    sys.path.insert(0, str(root))

    try:
        from crewai import Agent, Crew, Process, Task, LLM
        from src.agentic_rag.tools.custom_tool import DocumentSearchTool
    except Exception as e:
        return Check("PDF QA", False, f"import failed: {e}")

    query = _pick_pdf_query(pdf_path)

    try:
        pdf_tool = DocumentSearchTool(file_path=str(pdf_path))
    except Exception as e:
        return Check("PDF QA", False, f"DocumentSearchTool init failed: {e}")

    ok, status_msg = _wait_groundx_complete(pdf_tool, timeout_s=timeout_s)
    if not ok:
        return Check("PDF QA", False, status_msg)

    llm = LLM(
        model=os.getenv("SMOKE_PDF_LLM_MODEL", "ollama/deepseek-r1:7b"),
        base_url=os.getenv("SMOKE_OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    retriever = Agent(
        role="PDF evidence retriever for query: {query}",
        goal=(
            "Use ONLY the provided PDF search tool to retrieve verbatim evidence that answers the user query. "
            "Return evidence only. Do not answer the question. Do not use web or general knowledge."
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
            "If evidence is missing or insufficient, respond exactly: Not in document."
        ),
        backstory="You never use outside knowledge. You never invent quotes, citations, or page numbers.",
        verbose=False,
        llm=llm,
    )

    retrieval_task = Task(
        description=(
            "Retrieve ONLY verbatim evidence from the uploaded PDF for the user query: {query}. "
            "Do NOT use general knowledge. Do NOT answer the question. Output must be JSON only."
        ),
        expected_output='{"evidence":[{"page":<int >=1>,"quote":"<verbatim text from PDF>"}]}',
        agent=retriever,
    )

    response_task = Task(
        description=(
            "Using ONLY the JSON evidence from the retrieval task, answer the user query: {query}. "
            "Rules: (1) Use only evidence[].quote. (2) Cite every claim with (p.#) using evidence[].page. "
            "(3) If evidence is empty/insufficient, output exactly: Not in document."
        ),
        expected_output="A concise answer grounded in the PDF with (p.#) citations, or exactly: Not in document.",
        agent=synthesizer,
        context=[retrieval_task],
    )

    crew = Crew(
        agents=[retriever, synthesizer],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        crew_result = crew.kickoff(inputs={"query": query})
    except Exception as e:
        return Check("PDF QA", False, f"crew kickoff failed: {e}")

    tasks_output = getattr(crew_result, "tasks_output", None)
    if not tasks_output or len(tasks_output) < 1:
        return Check("PDF QA", False, "no task outputs returned by crew")

    retrieval_raw = getattr(tasks_output[0], "raw", None) or getattr(tasks_output[0], "output", None)
    if retrieval_raw is None:
        retrieval_raw = str(tasks_output[0])

    evidence = _parse_evidence_json(str(retrieval_raw))
    ok, why = _evidence_ok(evidence)
    if not ok:
        return Check("PDF QA", False, f"invalid evidence JSON: {why}")

    final_raw = None
    if len(tasks_output) >= 2:
        final_raw = getattr(tasks_output[1], "raw", None) or getattr(tasks_output[1], "output", None)
    if not isinstance(final_raw, str) or not final_raw.strip():
        final_raw = getattr(crew_result, "raw", None) or str(crew_result)

    if "not in document" in str(final_raw).strip().lower():
        return Check("PDF QA", False, f"unexpected refusal for answerable query: {query!r}")

    return Check("PDF QA", True, f"ok (query={query!r}, ingest={status_msg})")


def check_mcp_stdio(timeout_s: int) -> Check:
    if not os.getenv("LINKUP_API_KEY"):
        return Check("MCP Deep Research", False, "LINKUP_API_KEY is missing")

    root = _repo_root()
    mcp_project = (root.parent / "Multi-Agent-deep-researcher-mcp-windows-linux").resolve()
    server_py = mcp_project / "server.py"
    server_python = mcp_project / ".venv" / "bin" / "python"

    if not server_py.exists():
        return Check("MCP Deep Research", False, f"MCP server not found: {server_py}")
    if not server_python.exists():
        return Check("MCP Deep Research", False, f"MCP python not found: {server_python}")

    # Ensure CrewAI can write its sqlite cache in restricted environments.
    home_dir = (root / ".cache" / "mcp_home").resolve()
    home_dir.mkdir(parents=True, exist_ok=True)

    server_env = {
        "RICH_DISABLE": "1",
        "NO_COLOR": "1",
        "TERM": "dumb",
        "CLICOLOR": "0",
        "PYTHONUNBUFFERED": "1",
        "CREWAI_VERBOSE": "0",
        "HOME": str(home_dir),
        "LINKUP_API_KEY": os.environ["LINKUP_API_KEY"],
    }

    try:
        import anyio
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession
    except Exception as e:
        return Check("MCP Deep Research", False, f"mcp client import failed: {e}")

    async def _run() -> tuple[bool, str]:
        params = StdioServerParameters(command=str(server_python), args=[str(server_py)], env=server_env)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
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

                # Ask explicitly for a URL so we can validate basic web research behavior.
                query = "Provide 1 reputable https URL about climate change impacts (include the full URL)."
                with anyio.fail_after(timeout_s):
                    result = await session.call_tool(tool_name, {"query": query})

                content = getattr(result, "content", None)
                text = None
                if isinstance(content, list) and content:
                    first = content[0]
                    text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
                if not isinstance(text, str):
                    text = str(content if content is not None else result)

                urls = _extract_urls(text)
                if len(urls) < 1:
                    return False, f"tool call returned no URL(s); tools={names}"

                return True, f"ok (tools={names}, url={urls[0]})"

    try:
        ok, detail = anyio.run(_run)
        return Check("MCP Deep Research", ok, detail)
    except Exception as e:
        return Check("MCP Deep Research", False, f"exception: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for PDF QA + MCP Deep Research")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a local PDF to use")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout seconds per stage")
    args = parser.parse_args()

    root = _repo_root()
    pdf_path = Path(args.pdf).expanduser().resolve() if args.pdf else _default_pdf_path(root)
    if pdf_path is None:
        print("FAIL: No PDF provided and no default PDF found.", file=sys.stderr)
        return 2

    # Keep CrewAI quiet during smoke tests (stdout noise breaks MCP stdio in some setups).
    os.environ.setdefault("CREWAI_VERBOSE", "0")
    os.environ.setdefault("RICH_DISABLE", "1")
    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("TERM", "dumb")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    checks: list[Check] = [
        check_pdf_qa(pdf_path, timeout_s=args.timeout),
        check_mcp_stdio(timeout_s=args.timeout),
    ]

    all_ok = True
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"{status}: {c.name} - {c.detail}")
        all_ok = all_ok and c.ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
