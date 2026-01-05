from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class JsonlLogger:
    log_dir: Path = Path("logs")
    filename: str = "queries.jsonl"

    def _path(self) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / self.filename

    def log(self, entry: Mapping[str, Any]) -> None:
        payload = dict(entry)
        ts = float(payload.setdefault("ts", time.time()))
        payload.setdefault("timestamp", datetime.fromtimestamp(ts, tz=timezone.utc).isoformat())
        path = self._path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


_default_logger: Optional[JsonlLogger] = None


def get_logger() -> JsonlLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = JsonlLogger()
    return _default_logger
