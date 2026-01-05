from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


GROUNDX_API_KEY_ENV_VARS: tuple[str, ...] = (
    "GROUNDX_API_KEY",
    # Common fallbacks seen in older demos / different naming conventions.
    "EYELEVELAI_GROUNDX_API_KEY",
    "EYELEVEL_AI_GROUNDX_API_KEY",
)


@dataclass(frozen=True)
class KeyDiagnostics:
    checked_env_vars: tuple[str, ...]
    env_var_state: dict[str, dict[str, int | bool]]
    selected_env_var: Optional[str]
    key_present: bool
    masked_key: Optional[str]
    key_length: int

    def as_dict(self) -> dict[str, object]:
        return {
            "checked_env_vars": list(self.checked_env_vars),
            "env_var_state": self.env_var_state,
            "selected_env_var": self.selected_env_var,
            "key_present": self.key_present,
            "masked_key": self.masked_key,
            "key_length": self.key_length,
        }


def mask_key(k: str) -> str:
    s = (k or "").strip()
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}...{s[-4:]}"


def get_groundx_api_key() -> Optional[str]:
    for name in GROUNDX_API_KEY_ENV_VARS:
        v = os.getenv(name)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def groundx_key_diagnostics(api_key: Optional[str] = None) -> KeyDiagnostics:
    selected = None
    env_state: dict[str, dict[str, int | bool]] = {}
    for name in GROUNDX_API_KEY_ENV_VARS:
        v = os.getenv(name)
        v2 = v.strip() if isinstance(v, str) else ""
        env_state[name] = {"present": bool(v2), "length": len(v2)}

    if api_key is None:
        for name in GROUNDX_API_KEY_ENV_VARS:
            v = os.getenv(name)
            if isinstance(v, str) and v.strip():
                selected = name
                api_key = v.strip()
                break

    k = (api_key or "").strip()
    return KeyDiagnostics(
        checked_env_vars=GROUNDX_API_KEY_ENV_VARS,
        env_var_state=env_state,
        selected_env_var=selected,
        key_present=bool(k),
        masked_key=mask_key(k) if k else None,
        key_length=len(k),
    )
