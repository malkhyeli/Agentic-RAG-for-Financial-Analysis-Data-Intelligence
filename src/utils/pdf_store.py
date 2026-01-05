from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping, Optional


DEFAULT_CACHE_DIR = Path(".cache_uploaded_pdfs")
DEFAULT_META_PATH = DEFAULT_CACHE_DIR / "last.json"


@dataclass(frozen=True)
class StoredPdf:
    path: Path
    filename: str
    sha256: str
    size_bytes: int


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sanitize_filename(name: str) -> str:
    base = (name or "document.pdf").strip() or "document.pdf"
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    return base


def persist_pdf_bytes(
    data: bytes,
    filename: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> StoredPdf:
    """Write uploaded PDF bytes to a stable path and return metadata.

    The file is stored under `cache_dir` to avoid `TemporaryDirectory()` cleanup bugs.
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise ValueError("PDF data is empty")

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(bytes(data))
    safe_name = _sanitize_filename(filename)
    # Keep filenames manageable while staying collision-resistant.
    short = digest[:16]
    path = cache_dir / f"{short}_{safe_name}"
    if not path.exists():
        path.write_bytes(bytes(data))

    return StoredPdf(path=path, filename=safe_name, sha256=digest, size_bytes=len(data))


def ensure_pdf_in_session(
    session_state: MutableMapping[str, Any],
    *,
    data: bytes,
    filename: str,
    session_key: str = "pdf_store",
) -> StoredPdf:
    """Persist a PDF for the current Streamlit session and memoize in session_state."""
    stored: Optional[StoredPdf] = session_state.get(session_key)
    digest = sha256_bytes(data)
    if isinstance(stored, StoredPdf) and stored.sha256 == digest and stored.path.exists():
        return stored

    stored = persist_pdf_bytes(data, filename)
    session_state[session_key] = stored
    return stored


def write_last_meta(
    stored: StoredPdf,
    *,
    meta_path: Path = DEFAULT_META_PATH,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "path": str(stored.path),
        "name": stored.filename,
        "filename": stored.filename,
        "sha256": stored.sha256,
        # Backward compat for older scripts.
        "hash": stored.sha256,
        "size_bytes": stored.size_bytes,
    }
    if isinstance(extra, dict):
        for k, v in extra.items():
            if v is not None:
                payload[k] = v
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)


def read_last_meta(*, meta_path: Path = DEFAULT_META_PATH) -> Optional[dict[str, Any]]:
    try:
        if not meta_path.exists():
            return None
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_last_stored_pdf(*, meta_path: Path = DEFAULT_META_PATH) -> tuple[Optional[StoredPdf], Optional[dict[str, Any]]]:
    meta = read_last_meta(meta_path=meta_path)
    if not isinstance(meta, dict):
        return None, None
    path_val = meta.get("path")
    filename_val = meta.get("filename") or meta.get("name") or "document.pdf"
    sha = meta.get("sha256") or meta.get("hash")
    if not isinstance(path_val, str) or not path_val.strip():
        return None, meta
    path = Path(path_val).expanduser()
    if not path.exists():
        return None, meta
    if not isinstance(sha, str) or not sha.strip():
        try:
            sha = sha256_bytes(path.read_bytes())
        except Exception:
            return None, meta
    try:
        size = int(meta.get("size_bytes") or path.stat().st_size)
    except Exception:
        size = 0
    stored = StoredPdf(path=path.resolve(), filename=_sanitize_filename(str(filename_val)), sha256=str(sha), size_bytes=size)
    return stored, meta


def forget_last_meta(*, meta_path: Path = DEFAULT_META_PATH) -> None:
    try:
        if meta_path.exists():
            meta_path.unlink()
    except Exception:
        pass
