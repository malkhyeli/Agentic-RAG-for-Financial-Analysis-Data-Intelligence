from __future__ import annotations

import os
import re
import logging
import subprocess
import sys
import time
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from urllib.parse import urlparse

# URL shorteners are not acceptable as "citations" for deep research because they hide
# the real source and make verification harder.
_SHORTENER_DOMAINS = {
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "is.gd",
    "cutt.ly",
    "rebrand.ly",
    "shorturl.at",
    "lnkd.in",
}

def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _debug_enabled() -> bool:
    return _env_flag("MCP_DEBUG", "0")


def _repo_a_root() -> Path:
    # .../agentic_rag_deepseek/src/research/mcp_runner.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _mcp_home_dir() -> Path:
    override = os.getenv("MCP_RESEARCH_HOME")
    if isinstance(override, str) and override.strip():
        return Path(override.strip()).expanduser()
    return _repo_a_root() / ".cache" / "mcp_home"


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _mcp_errlog_path() -> Path:
    return _repo_a_root() / "logs" / "mcp_server.stderr.log"


def _tail_text(s: str, *, max_chars: int = 1200) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    # Keep error strings single-line for Streamlit debug and terminal hygiene.
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= max_chars:
        return t
    return "…" + t[-max_chars:]


class _DropExpectedMcpWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "Process group termination failed for PID" in msg:
            return False
        if "Process termination failed for PID" in msg:
            return False
        return True


_LOG_FILTER_INSTALLED = False


def _install_log_filter_once() -> None:
    global _LOG_FILTER_INSTALLED
    if _LOG_FILTER_INSTALLED:
        return
    try:
        logging.getLogger("mcp.os.posix.utilities").addFilter(_DropExpectedMcpWarnings())
    except Exception:
        pass
    _LOG_FILTER_INSTALLED = True


def _is_shortener_url(url: str) -> bool:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host in _SHORTENER_DOMAINS
    except Exception:
        return False


def _format_exception(e: Exception) -> str:
    """Make ExceptionGroup / TaskGroup errors actionable in logs/debug.

    Python 3.11+ uses ExceptionGroup/BaseExceptionGroup. anyio.TaskGroup failures often
    wrap the root cause in nested groups. This function recursively unwraps a few levels.
    """

    def _one_line(x: BaseException) -> str:
        try:
            msg = str(x)
        except Exception:
            msg = repr(x)
        msg = (msg or "").strip().replace("\n", " ")
        if len(msg) > 500:
            msg = msg[:500] + "…"
        return f"{type(x).__name__}: {msg}" if msg else f"{type(x).__name__}"

    def _flatten(x: BaseException, *, depth: int) -> list[str]:
        if depth <= 0:
            return [_one_line(x)]
        sub = getattr(x, "exceptions", None)  # ExceptionGroup/BaseExceptionGroup
        if isinstance(sub, (list, tuple)) and sub:
            out: list[str] = []
            for child in sub[:8]:
                if isinstance(child, BaseException):
                    out.extend(_flatten(child, depth=depth - 1))
                else:
                    out.append(repr(child))
            return out
        return [_one_line(x)]

    parts = _flatten(e, depth=3)
    # Always include the top-level message first, then a few flattened causes.
    head = _one_line(e)
    tail_parts = [p for p in parts if p and p != head]
    tail = "; ".join(tail_parts[:8])
    return head if not tail else f"{head} | causes: {tail}"


# Allow (broken) URLs with spaces inside the parentheses; we will repair them in _normalize_url.
URL_RE = re.compile(r"https?://[^\s\]\)\"']+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)\"']+)\)", re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


@dataclass(frozen=True)
class ResearchResult:
    summary: str
    citations: list[Citation]
    sources: list[str]
    raw: str
    latency_s: float
    error: Optional[str] = None
    # Source binding validation fields
    source_binding_valid: bool = True
    unbound_urls: list[str] = None  # type: ignore[assignment]
    bound_answer: Optional[str] = None  # Answer with sources properly bound

    def __post_init__(self) -> None:
        # Ensure unbound_urls defaults to empty list
        if self.unbound_urls is None:
            object.__setattr__(self, "unbound_urls", [])


def _normalize_url(url: str) -> str:
    """Normalize and lightly repair URLs extracted from model text.

    Models sometimes emit URLs with accidental spaces in the path, e.g.
    `https://example.com/foo bar/baz.pdf`. We repair this by joining the
    whitespace-separated pieces with `/`.

    We do not fetch or validate URLs here; we only clean and de-duplicate.
    """
    if not isinstance(url, str):
        return ""

    u = url.strip()
    # If the model emits a bare domain/path (missing scheme), assume https.
    if not u.lower().startswith(("http://", "https://")) and ("/" in u or "." in u):
        # Avoid turning plain text into URLs too aggressively.
        if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", u):
            u = "https://" + u
    # Strip common trailing punctuation that appears in prose.
    u = u.rstrip(".,;:!?")

    # Repair: collapse internal whitespace for URLs that start with http(s).
    # Example: "https://www.unesco.org/water stewardship/x.pdf" ->
    #          "https://www.unesco.org/water/stewardship/x.pdf"
    if ("http://" in u or "https://" in u) and any(ch.isspace() for ch in u):
        parts = [p for p in re.split(r"\s+", u) if p]
        if parts:
            base = parts[0].rstrip("/")
            rest = [p.strip("/") for p in parts[1:]]
            if rest:
                u = base + "/" + "/".join(rest)
            else:
                u = base

    return u


def _extract_urls(text: str) -> list[str]:
    """Extract URLs from free-form text.

    Note: this does not fetch or validate URLs; it only extracts and de-duplicates.
    """
    if not isinstance(text, str):
        return []

    urls = [_normalize_url(u) for u in URL_RE.findall(text)]

    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _is_homepage_url(url: str) -> bool:
    """Heuristic: treat bare domains and locale roots as homepages."""
    try:
        p = urlparse(url)
    except Exception:
        return True

    path = (p.path or "/").strip()
    if path in {"", "/"}:
        return True

    # Locale roots like /en, /en/, /en-us/
    lower = path.lower()
    if re.fullmatch(r"/[a-z]{2}(-[a-z]{2})?/?", lower):
        return True

    return False


def _filter_urls(urls: list[str], *, require_deep_links: bool) -> list[str]:
    """Filter URLs according to research-mode requirements."""
    if not urls:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = _normalize_url(u)
        if not u:
            continue
        if _is_shortener_url(u):
            continue
        if require_deep_links and _is_homepage_url(u):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _extract_citations(text: str, *, require_deep_links: bool) -> list[Citation]:
    """Extract citations from text.

    Prefers markdown link titles when present: [Title](https://...).
    Falls back to plain URL extraction.
    """
    if not isinstance(text, str):
        return []

    citations: list[Citation] = []
    seen: set[str] = set()

    # 1) Markdown links with titles.
    for m in MARKDOWN_LINK_RE.finditer(text):
        title = (m.group(1) or "").strip() or None
        url = _normalize_url(m.group(2) or "")
        if not url:
            continue
        if _is_shortener_url(url):
            continue
        if require_deep_links and _is_homepage_url(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        citations.append(Citation(url=url, title=title))

    # 2) Any remaining URLs.
    for url in _filter_urls(_extract_urls(text), require_deep_links=require_deep_links):
        if url in seen:
            continue
        seen.add(url)
        citations.append(Citation(url=url))

    return citations


_ALLOWLISTED_DOMAINS = {
    # Climate / environment / major institutions (non-exhaustive; verification is the primary gate).
    "ipcc.ch",
    "wmo.int",
    "noaa.gov",
    "nasa.gov",
    "usgs.gov",
    "epa.gov",
    "un.org",
    "undp.org",
    "unesco.org",
    "worldbank.org",
    "oecd.org",
}

_ALLOWLISTED_DOMAIN_SUFFIXES = (
    ".gov",
    ".edu",
    ".int",
    ".ac.uk",
)

# URL verification cache: url -> (ok, ts)
_VERIFY_CACHE: dict[str, tuple[bool, float]] = {}


def _host_for_url(url: str) -> str:
    try:
        p = urlparse(url)
    except Exception:
        return ""
    host = (p.netloc or "").split("@")[-1].split(":")[0].strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_allowlisted_domain(url: str) -> bool:
    host = _host_for_url(url)
    if not host:
        return False
    if host in _ALLOWLISTED_DOMAINS:
        return True
    return any(host.endswith(suf) for suf in _ALLOWLISTED_DOMAIN_SUFFIXES)


def _is_acceptable_content_type(content_type: str | None) -> bool:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    return ct in {"text/html", "application/pdf"}


def _verify_url_live(url: str, *, timeout_s: float) -> bool:
    # Import lazily so this module stays importable even if httpx isn't installed.
    try:
        import httpx  # type: ignore
    except Exception:
        return False

    headers = {"User-Agent": "agentic-rag-deep-research/1.0"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
            try:
                r = client.head(url)
            except Exception:
                r = None

            # Some servers block HEAD; fall back to a streamed GET.
            if r is None or r.status_code in {400, 403, 405}:
                r = None

            if r is not None:
                if r.status_code != 200:
                    return False
                if _is_acceptable_content_type(r.headers.get("content-type")):
                    return True

            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    return False
                if _is_acceptable_content_type(resp.headers.get("content-type")):
                    return True
    except Exception:
        return False

    return False


def _verify_url_cached(url: str, *, timeout_s: float) -> bool:
    now = time.time()
    ttl_env = (os.getenv("MCP_VERIFY_CACHE_TTL_S") or "").strip()
    ttl_s = int(ttl_env) if ttl_env.isdigit() else 60 * 60

    cached = _VERIFY_CACHE.get(url)
    if cached is not None:
        ok, ts = cached
        if now - ts <= ttl_s:
            return ok

    ok = _verify_url_live(url, timeout_s=timeout_s)
    _VERIFY_CACHE[url] = (ok, now)
    return ok


def _enforce_citation_safety(
    citations: list[Citation],
    *,
    require_deep_links: bool,
    min_citations: int,
) -> tuple[list[Citation], str | None]:
    """Filter/verify citations so Deep Research never accepts hallucinated sources."""
    verify_urls = _env_flag("MCP_VERIFY_URLS", "1")
    verify_timeout_env = (os.getenv("MCP_VERIFY_TIMEOUT_S") or "").strip()
    try:
        verify_timeout_s = float(verify_timeout_env) if verify_timeout_env else 5.0
    except Exception:
        verify_timeout_s = 5.0

    raw_urls = [c.url for c in (citations or []) if isinstance(getattr(c, "url", None), str)]
    if not raw_urls:
        return [], "no_citations"

    verified: list[Citation] = []
    any_rejected_by_verification = False
    seen: set[str] = set()

    for c in citations:
        url = _normalize_url(getattr(c, "url", "") or "")
        if not url or not url.lower().startswith(("http://", "https://")):
            continue
        if _is_shortener_url(url):
            continue
        if require_deep_links and _is_homepage_url(url):
            continue
        if url in seen:
            continue

        ok = False
        if _is_allowlisted_domain(url):
            ok = True
        elif verify_urls:
            ok = _verify_url_cached(url, timeout_s=verify_timeout_s)
            if not ok:
                any_rejected_by_verification = True
        else:
            any_rejected_by_verification = True

        if ok:
            seen.add(url)
            verified.append(Citation(url=url, title=c.title, snippet=c.snippet))
            if len(verified) >= int(min_citations):
                break

    if len(verified) >= int(min_citations):
        return verified, None
    if any_rejected_by_verification:
        return [], "verification_failed"
    return [], "no_citations"


def _is_list_tools_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if q in {"list_tools", "tools", "list mcp tools", "mcp tools"}:
        return True
    return ("mcp" in q and "tool" in q and any(k in q for k in ["list", "show", "available"]))


def _default_repo_b_path() -> Path:
    override = os.getenv("MCP_RESEARCH_REPO_B_PATH")
    if isinstance(override, str) and override.strip():
        return Path(override.strip())
    # Repo A lives at .../agentic_rag_deepseek; Repo B is a sibling by default.
    return Path(__file__).resolve().parents[3] / "Multi-Agent-deep-researcher-mcp-windows-linux"


def _repo_b_python(repo_b: Path) -> Path:
    """Pick the Python interpreter to run Repo B.

    Priority:
    1) MCP_RESEARCH_PYTHON override (explicit interpreter path).
       - If it points to a directory, we try <dir>/bin/python and <dir>/Scripts/python.exe.
    2) MCP_RESEARCH_USE_CALLER_PYTHON=1 to force using the current interpreter.
    3) Repo B local venv at <repo_b>/.venv/bin/python (if present).
    4) Fallback to the current interpreter.
    """

    override = os.getenv("MCP_RESEARCH_PYTHON")
    if isinstance(override, str) and override.strip():
        p = Path(override.strip())
        # Allow passing a venv directory instead of the python executable.
        if p.exists() and p.is_dir():
            cand = p / "bin" / "python"
            if cand.exists():
                return cand
            cand = p / "Scripts" / "python.exe"
            if cand.exists():
                return cand
        if p.exists():
            return p

    use_caller = (os.getenv("MCP_RESEARCH_USE_CALLER_PYTHON") or "").strip().lower()
    if use_caller in {"1", "true", "yes", "y", "on"}:
        return Path(sys.executable)

    venv_py = repo_b / ".venv" / "bin" / "python"
    if venv_py.exists():
        return venv_py

    return Path(sys.executable)


def _server_path(repo_b: Path) -> Path:
    override = os.getenv("MCP_RESEARCH_SERVER")
    if isinstance(override, str) and override.strip():
        return Path(override.strip())
    return repo_b / "server.py"


def _quiet_env(base: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = dict(base or os.environ)
    # Keep stdout clean for MCP stdio: anything printed to stdout breaks JSON-RPC parsing.
    env.setdefault("RICH_DISABLE", "1")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("CLICOLOR", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("CREWAI_VERBOSE", "0")
    # Reduce FastMCP chatter in stderr logs.
    env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")
    # Pass through search provider keys if set in Repo A .env / shell.
    if os.getenv("LINKUP_API_KEY"):
        env["LINKUP_API_KEY"] = os.getenv("LINKUP_API_KEY", "")
    if os.getenv("SERPER_API_KEY"):
        env["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
    if os.getenv("RESEARCH_SEARCH_PROVIDER"):
        env["RESEARCH_SEARCH_PROVIDER"] = os.getenv("RESEARCH_SEARCH_PROVIDER", "")

    # Unify LLM configuration between Repo A (this app) and Repo B (deep researcher).
    # Repo A uses DOCQA_* / OLLAMA_* env vars; Repo B expects CREWAI_*.
    if not (env.get("CREWAI_MODEL") or "").strip():
        docqa_model = (env.get("DOCQA_LLM_MODEL") or "").strip()
        if docqa_model:
            env["CREWAI_MODEL"] = docqa_model
    if not (env.get("CREWAI_BASE_URL") or "").strip():
        base_url = (env.get("OLLAMA_BASE_URL") or "").strip()
        if base_url:
            env["CREWAI_BASE_URL"] = base_url

    # Debug bridging: when MCP debug is enabled, also enable Repo B worker stderr logs.
    if _debug_enabled():
        env.setdefault("CREW_RESEARCH_DEBUG", "1")
    # Ensure Repo B is importable when invoked via subprocess with -c.
    # (Repo B is also set as cwd, but PYTHONPATH makes this robust.)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    return env


def _call_repo_b_subprocess(query: str, *, repo_b: Path, python_path: Path, timeout_s: int) -> tuple[str, Optional[str]]:
    """Fallback path when the MCP client is unavailable."""
    code = (
        "from agents import run_research\n"
        "import sys\n"
        "q = sys.argv[1]\n"
        "print(run_research(q))\n"
    )
    env = _quiet_env()
    # Ensure CrewAI SQLite storage lives in a writable location.
    home_dir = _mcp_home_dir()
    _ensure_dir(home_dir)
    env["HOME"] = str(home_dir)
    # Prepend Repo B to PYTHONPATH so `from agents import ...` always works.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_b) + (os.pathsep + existing_pp if existing_pp else "")
    # Repo B may run multiple agent steps and local LLM calls; allow longer here.
    # Do not artificially cap timeouts too low. Allow callers to control via timeout_s,
    # with an optional safety cap set by MCP_SUBPROCESS_HARD_TIMEOUT_S.
    _cap_env = os.getenv("MCP_SUBPROCESS_HARD_TIMEOUT_S", "").strip()
    _cap = int(_cap_env) if _cap_env.isdigit() else None
    hard_timeout = int(timeout_s)
    if _cap is not None:
        hard_timeout = min(hard_timeout, _cap)
    hard_timeout = max(10, hard_timeout)
    try:
        if _debug_enabled():
            print(f"[mcp_runner] subprocess fallback start (timeout={hard_timeout}s)", file=sys.stderr)
        proc = subprocess.run(
            [str(python_path), "-c", code, query],
            cwd=str(repo_b),
            env=env,
            capture_output=True,
            text=True,
            timeout=hard_timeout,
        )
        # Provide clear, actionable errors when Repo B is missing dependencies.
        stderr_txt = (proc.stderr or "").strip()
        stdout_txt = (proc.stdout or "").strip()
        combined = (stderr_txt + "\n" + stdout_txt).lower()
        if "no module named 'linkup'" in combined or 'no module named "linkup"' in combined:
            return "NO_SOURCES", "missing_dependency:linkup-sdk"
    except subprocess.TimeoutExpired as e:
        # e.stdout/stderr may be bytes even with text=True when timeout occurs
        stdout_part = (e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        stderr_part = (e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        partial = (stdout_part + "\n" + stderr_part).strip()
        partial = partial[:800] if partial else ""
        return "NO_SOURCES", f"subprocess_timeout_s={hard_timeout}: {partial}"
    except Exception as e:
        return "NO_SOURCES", f"subprocess_failed: {e}"

    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        return ("NO_SOURCES" if not out else out), f"subprocess_rc={proc.returncode}: {err[:400]}"
    return out, None


async def _call_mcp_stdio(
    query: str,
    *,
    repo_b: Path,
    python_path: Path,
    server_py: Path,
    timeout_s: int,
) -> tuple[str, Optional[str]]:
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as e:
        return "NO_SOURCES", f"mcp_import_failed: {e}"

    _install_log_filter_once()
    env = _quiet_env()
    # Ensure CrewAI SQLite storage lives in a writable location.
    home_dir = _mcp_home_dir()
    _ensure_dir(home_dir)
    env["HOME"] = str(home_dir)
    server = StdioServerParameters(
        command=str(python_path),
        args=[str(server_py)],
        env=env,
        cwd=str(repo_b),
    )

    # MCP client versions differ on whether `read_timeout_seconds` expects a `datetime.timedelta`
    # or a numeric seconds value. We build both and pick the compatible one at call time.
    read_timeout_seconds = max(5, int(timeout_s))
    read_timeout_arg: object = datetime.timedelta(seconds=read_timeout_seconds)
    errlog_fh = None
    errlog_start = None
    errlog: object = sys.stderr
    try:
        errlog_path = _mcp_errlog_path()
        _ensure_dir(errlog_path.parent)
        errlog_fh = open(errlog_path, "a+", encoding="utf-8")
        errlog_fh.seek(0, os.SEEK_END)
        errlog_start = errlog_fh.tell()
        errlog = errlog_fh
    except Exception:
        errlog_fh = None
        errlog_start = None
        errlog = sys.stderr

    try:
        async with stdio_client(server, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                if _is_list_tools_query(query):
                    tools_resp = await session.list_tools()
                    tools = getattr(tools_resp, "tools", None) or tools_resp
                    names: list[str] = []
                    if isinstance(tools, list):
                        for t in tools:
                            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                            if name:
                                names.append(name)
                    if not names:
                        return "NO_SOURCES", "no_tools"
                    formatted = "MCP tools available:\n" + "\n".join([f"- {n}" for n in names])
                    return formatted, None

                try:
                    result = await session.call_tool(
                        "crew_research",
                        {"query": query},
                        read_timeout_seconds=read_timeout_arg,  # type: ignore[arg-type]
                    )
                except TypeError:
                    # Back-compat: older MCP client expects a numeric seconds value.
                    result = await session.call_tool(
                        "crew_research",
                        {"query": query},
                        read_timeout_seconds=read_timeout_seconds,  # type: ignore[arg-type]
                    )
                if getattr(result, "isError", False):
                    # Treat tool errors as recoverable so the caller can fall back.
                    return "NO_SOURCES", "mcp_failed: mcp_tool_error"

                content = getattr(result, "content", None)
                parts: list[str] = []
                if isinstance(content, list) and content:
                    for c in content:
                        txt = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt)
                text = "\n\n".join(parts) if parts else str(content if content is not None else result)
                text = text.strip()

                # Some MCP servers (including our Repo B) fail closed by returning the sentinel
                # string "NO_SOURCES" instead of raising an MCP tool error. When that happens,
                # surface the stderr tail so the caller can see why it failed (missing keys,
                # network errors, LLM errors, etc.).
                if text.upper() in {"NO_SOURCES", "NO_SOURCE", "NOSOURCES"}:
                    tail = ""
                    if errlog_fh is not None and errlog_start is not None:
                        try:
                            errlog_fh.flush()
                            errlog_fh.seek(errlog_start)
                            tail = _tail_text(errlog_fh.read())
                        except Exception:
                            tail = ""
                    msg = f"no_sources: {tail}" if tail else "no_sources"
                    return "NO_SOURCES", msg

                return (text or "NO_SOURCES"), None
    except Exception as e:
        tail = ""
        if errlog_fh is not None and errlog_start is not None:
            try:
                errlog_fh.flush()
                errlog_fh.seek(errlog_start)
                tail = _tail_text(errlog_fh.read())
            except Exception:
                tail = ""
        msg = _format_exception(e)
        if tail:
            msg = f"{msg} | server_stderr_tail: {tail}"
        return "NO_SOURCES", f"mcp_failed: {msg}"
    finally:
        if errlog_fh is not None:
            try:
                errlog_fh.close()
            except Exception:
                pass


def run_mcp_research(
    query: str,
    *,
    repo_b_path: Optional[Path] = None,
    timeout_s: int = 180,
    min_citations: int = 1,
    require_deep_links: bool = True,
    max_retries: int = 2,
) -> ResearchResult:
    """Run deep web research via Repo B MCP stdio server (with subprocess fallback).

    Reliability rules:
    - If the tool returns zero citations (after filtering), retry up to `max_retries` times.
    - If `require_deep_links` is True, homepage-like URLs are discarded.
    """
    started = time.time()
    has_linkup = bool((os.getenv("LINKUP_API_KEY") or "").strip())
    has_serper = bool((os.getenv("SERPER_API_KEY") or "").strip())
    if not has_linkup and not has_serper:
        return ResearchResult(
            summary="NO_SOURCES",
            citations=[],
            sources=[],
            raw="Missing search API key. Set SERPER_API_KEY (recommended) or LINKUP_API_KEY.",
            latency_s=time.time() - started,
            error="missing_key:SERPER_API_KEY_or_LINKUP_API_KEY",
        )
    repo_b = repo_b_path or _default_repo_b_path()
    if not repo_b.exists():
        return ResearchResult(
            summary="NO_SOURCES",
            citations=[],
            sources=[],
            raw="Repo B not found.",
            latency_s=time.time() - started,
            error="repo_b_missing",
        )

    python_path = _repo_b_python(repo_b)
    server_py = _server_path(repo_b)
    if not server_py.exists():
        return ResearchResult(
            summary="NO_SOURCES",
            citations=[],
            sources=[],
            raw=f"Missing server: {server_py}",
            latency_s=time.time() - started,
            error="server_missing",
        )

    raw_text: str = ""
    err: Optional[str] = None

    # Keep the overall call within timeout_s by splitting time across attempts.
    attempts = max(1, int(max_retries) + 1)
    per_attempt_timeout = max(10, int(timeout_s // attempts))

    last_raw: str = "NO_SOURCES"
    last_err: Optional[str] = None
    last_safety_err: Optional[str] = None

    for attempt in range(attempts):
        q = query
        # If we are retrying, make the requirement explicit to the researcher.
        if attempt > 0:
            q = (
                query
                + "\n\nReturn ONLY direct, fully-qualified URLs to specific report pages or PDFs. "
                + "No URL shorteners (bit.ly, t.co, etc). No homepages. "
                + "Provide at least "
                + str(int(min_citations))
                + " distinct sources."
            )

        try:
            import anyio
            import functools

            runner = functools.partial(
                _call_mcp_stdio,
                q,
                repo_b=repo_b,
                python_path=python_path,
                server_py=server_py,
                timeout_s=per_attempt_timeout,
            )
            raw_text, err = anyio.run(runner)

            # If MCP stdio returns a TaskGroup/ExceptionGroup style failure, fall back to subprocess.
            if isinstance(err, str) and err.startswith("mcp_failed:"):
                try:
                    if _debug_enabled():
                        print(f"[mcp_runner] MCP stdio failed, falling back to subprocess: {err}", file=sys.stderr)
                    else:
                        print("[mcp_runner] MCP stdio failed; using subprocess fallback", file=sys.stderr)
                except Exception:
                    pass
                raw_text, err = _call_repo_b_subprocess(
                    q, repo_b=repo_b, python_path=python_path, timeout_s=per_attempt_timeout
                )
        except Exception:
            raw_text, err = _call_repo_b_subprocess(
                q, repo_b=repo_b, python_path=python_path, timeout_s=per_attempt_timeout
            )

        raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)
        raw_text = raw_text.strip() or "NO_SOURCES"

        extracted = _extract_citations(raw_text, require_deep_links=require_deep_links)
        citations, safety_err = _enforce_citation_safety(
            extracted,
            require_deep_links=require_deep_links,
            min_citations=int(min_citations),
        )
        sources = [c.url for c in citations]

        last_raw = raw_text
        last_err = err
        last_safety_err = safety_err
        # Fail closed with a specific error if Repo B is missing required dependencies.
        if err == "missing_dependency:linkup-sdk":
            return ResearchResult(
                summary="NO_SOURCES",
                citations=[],
                sources=[],
                raw=last_raw,
                latency_s=time.time() - started,
                error=err,
            )

        # Success condition: enough citations after filtering.
        if len(citations) >= int(min_citations) and raw_text != "NO_SOURCES":
            return ResearchResult(
                summary=raw_text,
                citations=citations,
                sources=sources,
                raw=raw_text,
                latency_s=time.time() - started,
                error=err,
            )

        # If MCP returned a hard error, do not spin on retries.
        if isinstance(err, str) and err.startswith("mcp_failed: mcp_tool_error"):
            break

    # Failed all attempts.
    extracted = _extract_citations(last_raw, require_deep_links=require_deep_links)
    citations, safety_err = _enforce_citation_safety(
        extracted,
        require_deep_links=require_deep_links,
        min_citations=int(min_citations),
    )
    sources = [c.url for c in citations]
    final_code = safety_err or last_safety_err or "no_citations"
    final_err = final_code
    if last_err and last_err != final_code:
        final_err = f"{final_code}: {last_err}"
    return ResearchResult(
        summary="NO_SOURCES",
        citations=[],
        sources=[],
        raw=last_raw,
        latency_s=time.time() - started,
        error=final_err,
        source_binding_valid=False,
        unbound_urls=[],
        bound_answer=None,
    )


# =====================================================
# Source Binding Validation
# =====================================================

# Number words for parsing requested source counts
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}


def parse_requested_source_count(query: str) -> Optional[int]:
    """Parse the number of sources requested in the query.

    Detects patterns like:
    - "give me 3 sources"
    - "provide three sources"
    - "5 authoritative sources"
    - "at least 2 sources"

    Returns None if no specific count is requested.
    """
    if not isinstance(query, str):
        return None

    q_lower = query.lower()

    # Pattern: "<number> sources" or "<number> authoritative sources"
    patterns = [
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:authoritative\s+)?(?:verified\s+)?(?:reliable\s+)?sources?",
        r"(?:give|provide|list|show|find|get)\s+(?:me\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+",
        r"(?:at\s+least|minimum|min)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sources?",
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:different\s+)?(?:web\s+)?sources?",
    ]

    for pattern in patterns:
        match = re.search(pattern, q_lower)
        if match:
            num_str = match.group(1).strip()
            if num_str in _NUMBER_WORDS:
                return _NUMBER_WORDS[num_str]

    return None


def build_citation_only_answer(
    citations: list[Citation],
    query: str,
    *,
    max_citations: int = 10,
) -> str:
    """Build an answer using ONLY verified citations - no model-generated text.

    This creates a structured answer where each source is listed with:
    - Title (from citation or derived from URL)
    - URL (verified)
    - Snippet (if available)

    This completely avoids hallucinated source names by not using any model text.

    Args:
        citations: List of verified Citation objects.
        query: The original query (for context in the header).
        max_citations: Maximum number of citations to include.

    Returns:
        Formatted answer with only verified sources.
    """
    if not citations:
        return "No verified sources found for this query."

    # Limit to max_citations
    citations_to_use = citations[:max_citations]
    n = len(citations_to_use)

    # Build header
    header = f"Found {n} verified source{'s' if n != 1 else ''} for your query:\n"

    # Build each source entry
    source_entries: list[str] = []
    for idx, c in enumerate(citations_to_use, 1):
        # Use title from citation, or derive from URL
        title = c.title
        if not title or title.lower() in {"untitled", "source", "link"}:
            # Extract domain and path for a meaningful title
            try:
                parsed = urlparse(c.url)
                domain = parsed.netloc.replace("www.", "")
                path = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
                if path and len(path) > 3:
                    # Clean up path to be readable
                    path = path.replace("-", " ").replace("_", " ")
                    path = re.sub(r"\.(html?|php|aspx?|pdf)$", "", path, flags=re.IGNORECASE)
                    title = f"{domain}: {path[:50]}"
                else:
                    title = domain
            except Exception:
                title = f"Source {idx}"

        entry = f"**[{idx}] {title}**\n"
        entry += f"   URL: {c.url}\n"
        if c.snippet:
            # Truncate snippet if too long
            snippet = c.snippet[:200] + "..." if len(c.snippet) > 200 else c.snippet
            entry += f"   Summary: {snippet}\n"

        source_entries.append(entry)

    return header + "\n" + "\n".join(source_entries)


# Common hallucinated source patterns that LLMs often produce
_HALLUCINATED_SOURCE_PATTERNS = [
    r"\bIPCC\s+AR\d+\b",  # "IPCC AR5", "IPCC AR6"
    r"\bIMB\s+Study\b",
    r"\bUniversity\s+report\b",
    r"\bUN\s+Report\b",
    r"\bWorld\s+Bank\s+Report\b",
    r"\bNASA\s+Study\b",
    r"\bNOAA\s+Report\b",
    r"\bScientific\s+Study\b",
    r"\bResearch\s+Paper\b",
    r"\bJournal\s+Article\b",
    r"\bAccording\s+to\s+(?:the\s+)?(?:latest\s+)?(?:recent\s+)?studies?\b",
    r"\bExperts\s+(?:say|report|confirm)\b",
]

_HALLUCINATED_PATTERN_RE = re.compile(
    "|".join(_HALLUCINATED_SOURCE_PATTERNS), re.IGNORECASE
)


def validate_source_binding(
    summary: str,
    verified_urls: list[str],
    *,
    strict: bool = True,
) -> tuple[bool, list[str], str]:
    """Validate that all URLs in the summary are in the verified set.

    Args:
        summary: The raw summary text from the LLM.
        verified_urls: List of verified URLs from citation safety checks.
        strict: If True, fail if any unverified URL is found.

    Returns:
        Tuple of (is_valid, unbound_urls, sanitized_summary).
        - is_valid: True if all URLs in summary are verified.
        - unbound_urls: List of URLs in summary that are not in verified_urls.
        - sanitized_summary: Summary with unverified URLs removed (if not strict).
    """
    if not isinstance(summary, str) or not summary.strip():
        return True, [], ""

    # Extract all URLs from the summary
    summary_urls = _extract_urls(summary)

    # Normalize verified URLs for comparison
    verified_set = {_normalize_url(u).lower().rstrip("/") for u in verified_urls}

    # Find unbound URLs (URLs in summary but not in verified set)
    unbound: list[str] = []
    for url in summary_urls:
        normalized = _normalize_url(url).lower().rstrip("/")
        if normalized not in verified_set:
            unbound.append(url)

    # If strict mode and there are unbound URLs, fail
    if strict and unbound:
        return False, unbound, summary

    # Sanitize: remove unbound URLs from summary
    sanitized = summary
    for url in unbound:
        sanitized = sanitized.replace(url, "[source removed]")

    return len(unbound) == 0, unbound, sanitized


def detect_hallucinated_sources(summary: str) -> list[str]:
    """Detect common hallucinated source patterns in the summary.

    Returns a list of matched patterns that look like hallucinated sources.
    """
    if not isinstance(summary, str):
        return []

    matches = _HALLUCINATED_PATTERN_RE.findall(summary)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            result.append(m)
    return result


def build_bound_answer(
    summary: str,
    citations: list[Citation],
    *,
    max_citations: int = 10,
) -> str:
    """Build an answer with sources properly bound to verified URLs.

    This creates a structured answer where:
    1. The main text is the summary (stripped of unverified content)
    2. Sources are listed with their verified URLs only

    Args:
        summary: The raw summary text.
        citations: List of verified Citation objects.
        max_citations: Maximum number of citations to include.

    Returns:
        Formatted answer with bound sources.
    """
    if not citations:
        return "No verified sources available."

    # Detect and warn about hallucinated sources
    hallucinated = detect_hallucinated_sources(summary)

    # Extract URLs from summary for validation
    summary_urls = set(_extract_urls(summary))
    verified_urls = {c.url for c in citations}

    # Remove URLs from summary that aren't in verified set
    cleaned_summary = summary
    for url in summary_urls:
        if url not in verified_urls:
            cleaned_summary = cleaned_summary.replace(url, "")

    # Remove hallucinated source mentions (they have no URL backing)
    for pattern in hallucinated:
        cleaned_summary = re.sub(
            re.escape(pattern) + r"[,;:\s]*",
            "",
            cleaned_summary,
            flags=re.IGNORECASE,
        )

    # Clean up whitespace
    cleaned_summary = re.sub(r"\s+", " ", cleaned_summary).strip()
    cleaned_summary = re.sub(r"\s*([,;:])\s*([,;:])", r"\1", cleaned_summary)

    # Build the sources section with only verified URLs
    sources_lines: list[str] = []
    for idx, c in enumerate(citations[:max_citations], 1):
        title = c.title or f"Source {idx}"
        sources_lines.append(f"[{idx}] {title}: {c.url}")

    sources_section = "\n".join(sources_lines) if sources_lines else ""

    # Combine cleaned summary with verified sources
    if sources_section:
        return f"{cleaned_summary}\n\nVerified Sources:\n{sources_section}"
    return cleaned_summary


def run_mcp_research_with_binding(
    query: str,
    *,
    repo_b_path: Optional[Path] = None,
    timeout_s: int = 180,
    min_citations: int = 1,
    require_deep_links: bool = True,
    max_retries: int = 2,
    fail_on_unbound: bool = True,
    requested_source_count: Optional[int] = None,
    use_citation_only_answer: bool = True,
) -> ResearchResult:
    """Run deep research with strict source binding validation (fail-closed).

    This wraps run_mcp_research and enforces:
    1. All URLs in answer must be verified (no hallucinated sources)
    2. If user requests N sources, exactly N must be returned or fail
    3. Answer is built ONLY from verified citations (no model-generated text)

    Args:
        query: Research query.
        repo_b_path: Path to Repo B (optional).
        timeout_s: Timeout in seconds.
        min_citations: Minimum required citations.
        require_deep_links: Require non-homepage URLs.
        max_retries: Max retry attempts.
        fail_on_unbound: If True, fail closed when unbound URLs are detected.
        requested_source_count: If set, enforce exactly this many sources.
        use_citation_only_answer: If True, build answer only from citations (no model text).

    Returns:
        ResearchResult with source binding validation fields populated.
    """
    # Parse requested source count from query if not explicitly provided
    if requested_source_count is None:
        requested_source_count = parse_requested_source_count(query)

    # If user requested specific count, use it as min_citations
    effective_min = max(min_citations, requested_source_count or 1)

    # Run the base research with higher candidate count for better filtering
    result = run_mcp_research(
        query,
        repo_b_path=repo_b_path,
        timeout_s=timeout_s,
        min_citations=effective_min,
        require_deep_links=require_deep_links,
        max_retries=max_retries,
    )

    # If no citations, return as-is (already failed closed)
    if not result.citations:
        error_msg = result.error or "no_citations"
        if requested_source_count:
            error_msg = f"Requested {requested_source_count} sources but found 0 verified sources. {error_msg}"
        return ResearchResult(
            summary=result.summary,
            citations=[],
            sources=[],
            raw=result.raw,
            latency_s=result.latency_s,
            error=error_msg,
            source_binding_valid=False,
            unbound_urls=[],
            bound_answer=None,
        )

    # Enforce requested source count
    actual_count = len(result.citations)
    if requested_source_count is not None and actual_count < requested_source_count:
        # Fail closed: not enough verified sources
        return ResearchResult(
            summary=result.summary,
            citations=result.citations,
            sources=result.sources,
            raw=result.raw,
            latency_s=result.latency_s,
            error=f"insufficient_sources: requested {requested_source_count}, found {actual_count} verified",
            source_binding_valid=False,
            unbound_urls=[],
            bound_answer=build_citation_only_answer(
                result.citations, query, max_citations=actual_count
            ),
        )

    # If user requested specific count, limit to that count
    citations_to_use = result.citations
    if requested_source_count is not None and actual_count > requested_source_count:
        citations_to_use = result.citations[:requested_source_count]

    # Build the citation-only answer (completely avoids hallucinated sources)
    if use_citation_only_answer:
        bound_answer = build_citation_only_answer(
            citations_to_use, query, max_citations=len(citations_to_use)
        )
    else:
        # Legacy behavior: use model text with cleaning
        bound_answer = build_bound_answer(result.summary, citations_to_use)

    # Validate source binding (for diagnostics, but we use citation-only answer anyway)
    verified_urls = [c.url for c in citations_to_use]
    is_valid, unbound, _ = validate_source_binding(
        result.summary, verified_urls, strict=fail_on_unbound
    )

    # Detect hallucinated sources in original summary (for logging)
    hallucinated = detect_hallucinated_sources(result.summary)

    # When using citation-only answer, source binding is always valid
    # because we don't use any model-generated text
    if use_citation_only_answer:
        return ResearchResult(
            summary=result.summary,
            citations=citations_to_use,
            sources=[c.url for c in citations_to_use],
            raw=result.raw,
            latency_s=result.latency_s,
            error=None,  # No error - we have valid citation-only answer
            source_binding_valid=True,  # Citation-only is always valid
            unbound_urls=unbound,  # Keep for diagnostics
            bound_answer=bound_answer,
        )

    # Legacy behavior with fail_on_unbound check
    if fail_on_unbound and (not is_valid or hallucinated):
        error_parts = []
        if not is_valid:
            error_parts.append(f"unbound_urls={unbound}")
        if hallucinated:
            error_parts.append(f"hallucinated_sources={hallucinated}")
        error_msg = "; ".join(error_parts)

        return ResearchResult(
            summary=result.summary,
            citations=citations_to_use,
            sources=[c.url for c in citations_to_use],
            raw=result.raw,
            latency_s=result.latency_s,
            error=f"source_binding_failed: {error_msg}",
            source_binding_valid=False,
            unbound_urls=unbound,
            bound_answer=bound_answer,
        )

    return ResearchResult(
        summary=result.summary,
        citations=citations_to_use,
        sources=[c.url for c in citations_to_use],
        raw=result.raw,
        latency_s=result.latency_s,
        error=result.error,
        source_binding_valid=is_valid and not hallucinated,
        unbound_urls=unbound,
        bound_answer=bound_answer,
    )
