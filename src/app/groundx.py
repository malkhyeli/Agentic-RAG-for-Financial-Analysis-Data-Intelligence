from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from groundx import GroundX
from groundx.core.api_error import ApiError

from src.agentic_rag.tools.custom_tool import DocumentSearchTool, MissingApiKeyError
from src.utils.env import get_groundx_api_key, groundx_key_diagnostics


@dataclass(frozen=True)
class GroundXResult:
    ok: bool
    kind: str  # "ok" | "missing_key" | "api_error" | "unexpected" | "ingest_incomplete" | "empty" | "error" | "in_progress"
    message: str
    status_code: Optional[int] = None
    body: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "message": self.message,
            "status_code": self.status_code,
            "body": self.body,
            "diagnostics": self.diagnostics,
        }


def _clean_key(k: Optional[str]) -> str:
    return (k or "").strip()


def _diagnostics(api_key_override: Optional[str]) -> dict[str, Any]:
    override = _clean_key(api_key_override)
    diag = groundx_key_diagnostics(api_key=(override or None)).as_dict()
    diag["key_source"] = "override" if override else (diag.get("selected_env_var") or "none")
    return diag


def _user_message_for_api_error(status_code: Optional[int]) -> str:
    if status_code == 401:
        return "GroundX rejected the API key (401). Check that GROUNDX_API_KEY is correct and active."
    return f"GroundX error (status={status_code})."

def _normalize_status(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        s = str(val).strip()
    except Exception:
        return None
    s = s.lower()
    return s or None


def choose_groundx_key(api_key_override: Optional[str]) -> Optional[str]:
    override = _clean_key(api_key_override)
    if override:
        return override
    return get_groundx_api_key()


def test_groundx_connection(api_key_override: Optional[str]) -> GroundXResult:
    chosen_key = choose_groundx_key(api_key_override)
    if not chosen_key:
        return GroundXResult(
            ok=False,
            kind="missing_key",
            message="GROUNDX_API_KEY is not set. Add it to .env (recommended) or export it in your shell.",
            diagnostics=_diagnostics(api_key_override),
        )

    try:
        client = GroundX(api_key=chosen_key)
        client.buckets.list(n=1)
        return GroundXResult(ok=True, kind="ok", message="GroundX connection OK.", diagnostics=_diagnostics(api_key_override))
    except ApiError as e:
        status = getattr(e, "status_code", None)
        return GroundXResult(
            ok=False,
            kind="api_error",
            message=_user_message_for_api_error(status),
            status_code=status,
            body=getattr(e, "body", None),
            diagnostics=_diagnostics(api_key_override),
        )
    except Exception as e:
        return GroundXResult(
            ok=False,
            kind="unexpected",
            message=f"Unexpected error while testing GroundX: {e}",
            diagnostics=_diagnostics(api_key_override),
        )


def init_document_search_tool(file_path: str, api_key_override: Optional[str]) -> tuple[Optional[DocumentSearchTool], GroundXResult]:
    try:
        tool = DocumentSearchTool(
            file_path=file_path,
            api_key=(_clean_key(api_key_override) or None),
            # Never block Streamlit for minutes inside __init__. We poll status incrementally.
            ready_timeout_s=0,
        )
        diag = _diagnostics(api_key_override)
        diag.update(tool.debug_status())
        ready_docs = bool(getattr(tool, "document_ids", None))
        bucket_file_count_val = getattr(tool, "bucket_file_count", None)
        ready_bucket = False
        if bucket_file_count_val is not None:
            try:
                ready_bucket = float(bucket_file_count_val) > 0
            except Exception:
                ready_bucket = False

        if tool.ready and (ready_docs or ready_bucket):
            return tool, GroundXResult(ok=True, kind="ok", message="Indexed PDF.", diagnostics=diag)

        last_status_raw = diag.get("ingest_status") or diag.get("last_poll_status")
        last_status = _normalize_status(last_status_raw)
        if last_status and (last_status in {"error", "cancelled", "canceled"} or last_status.startswith("error:")):
            return (
                None,
                GroundXResult(
                    ok=False,
                    kind="error",
                    message=f"GroundX ingest failed (status: {last_status_raw}).",
                    diagnostics=diag,
                ),
            )

        message = f"GroundX ingest still in progress (status: {last_status_raw})."
        return (
            None,
            GroundXResult(
                ok=False,
                kind="in_progress" if tool.in_progress or last_status else "ingest_incomplete",
                message=message,
                diagnostics=diag,
            ),
        )
    except MissingApiKeyError as e:
        return (
            None,
            GroundXResult(
                ok=False,
                kind="missing_key",
                message=str(e),
                diagnostics=_diagnostics(api_key_override),
            ),
        )
    except FileNotFoundError as e:
        return (
            None,
            GroundXResult(
                ok=False,
                kind="file_missing",
                message=str(e),
                diagnostics=_diagnostics(api_key_override),
            ),
        )
    except ApiError as e:
        status = getattr(e, "status_code", None)
        return (
            None,
            GroundXResult(
                ok=False,
                kind="api_error",
                message=_user_message_for_api_error(status),
                status_code=status,
                body=getattr(e, "body", None),
                diagnostics=_diagnostics(api_key_override),
            ),
        )
    except Exception as e:
        return (
            None,
            GroundXResult(
                ok=False,
                kind="unexpected",
                message=f"Unexpected error while indexing PDF: {e}",
                diagnostics=_diagnostics(api_key_override),
            ),
        )


def test_pdf_retrieval(pdf_tool, query: str = "sea-level rise"):
    try:
        payload = pdf_tool.test_retrieval(query=query, n=3)
        has_results = bool(payload.get("results"))
        has_error = bool(payload.get("error"))
        kind = "ok" if has_results else ("error" if has_error else "empty")
        message = (
            "Retrieval returned results."
            if has_results
            else ("Retrieval error." if has_error else "Retrieval returned no results.")
        )
        return GroundXResult(
            ok=has_results,
            kind=kind,
            message=message,
            diagnostics={"retrieval": payload, **pdf_tool.debug_status()},
        )
    except ApiError as e:
        status = getattr(e, "status_code", None)
        return GroundXResult(
            ok=False,
            kind="api_error",
            message=_user_message_for_api_error(status),
            status_code=status,
            body=getattr(e, "body", None),
            diagnostics=pdf_tool.debug_status(),
        )
    except Exception as e:
        return GroundXResult(
            ok=False,
            kind="unexpected",
            message=f"Unexpected error while testing retrieval: {e}",
            diagnostics=pdf_tool.debug_status(),
        )


def resume_document_search_tool(
    bucket_id: int,
    process_id: str,
    api_key_override: Optional[str],
    document_ids: Optional[list[str]] = None,
) -> tuple[Optional[DocumentSearchTool], GroundXResult]:
    try:
        tool = DocumentSearchTool(
            file_path=None,
            bucket_id=bucket_id,
            process_id=process_id,
            document_ids=document_ids,
            api_key=(_clean_key(api_key_override) or None),
            # Never block Streamlit for minutes inside __init__. We poll status incrementally.
            ready_timeout_s=0,
        )
        diag = _diagnostics(api_key_override)
        diag.update(tool.debug_status())
        ready_docs = bool(getattr(tool, "document_ids", None))
        bucket_file_count_val = getattr(tool, "bucket_file_count", None)
        ready_bucket = False
        if bucket_file_count_val is not None:
            try:
                ready_bucket = float(bucket_file_count_val) > 0
            except Exception:
                ready_bucket = False

        if tool.ready and (ready_docs or ready_bucket):
            return tool, GroundXResult(ok=True, kind="ok", message="Indexed PDF.", diagnostics=diag)

        last_status_raw = diag.get("ingest_status") or diag.get("last_poll_status")
        last_status = _normalize_status(last_status_raw)
        if last_status and (last_status in {"error", "cancelled", "canceled"} or last_status.startswith("error:")):
            return (
                None,
                GroundXResult(
                    ok=False,
                    kind="error",
                    message=f"GroundX ingest failed (status: {last_status_raw}).",
                    diagnostics=diag,
                ),
            )

        return (
            None,
            GroundXResult(
                ok=False,
                kind="in_progress" if tool.in_progress or last_status else "ingest_incomplete",
                message=f"GroundX ingest still in progress (status: {last_status_raw}).",
                diagnostics=diag,
            ),
        )
    except MissingApiKeyError as e:
        return (
            None,
            GroundXResult(
                ok=False,
                kind="missing_key",
                message=str(e),
                diagnostics=_diagnostics(api_key_override),
            ),
        )
    except ApiError as e:
        status = getattr(e, "status_code", None)
        return (
            None,
            GroundXResult(
                ok=False,
                kind="api_error",
                message=_user_message_for_api_error(status),
                status_code=status,
                body=getattr(e, "body", None),
                diagnostics=_diagnostics(api_key_override),
            ),
        )
    except Exception as e:
        return (
            None,
            GroundXResult(
                ok=False,
                kind="unexpected",
                message=f"Unexpected error while resuming PDF indexing: {e}",
                diagnostics=_diagnostics(api_key_override),
            ),
        )
