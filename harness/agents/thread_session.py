from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThreadSessionManager:
    root: Path
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        safe = "".join(ch for ch in thread_id if ch.isalnum() or ch in ("-", "_")) or "default"
        return self.root / f"{safe}.json"

    def _load(self, thread_id: str) -> dict[str, Any]:
        path = self._path(thread_id)
        if not path.exists():
            return {"thread_id": thread_id, "updated_at": _utc_now()}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, thread_id: str, payload: dict[str, Any]) -> None:
        path = self._path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["thread_id"] = thread_id
        payload["updated_at"] = _utc_now()
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def get_state(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            return self._load(thread_id)

    def get_working_directory(self, thread_id: str) -> str | None:
        with self._lock:
            data = self._load(thread_id)
        value = str(data.get("working_directory", "") or "").strip()
        return value or None

    def set_working_directory(self, thread_id: str, working_directory: str) -> None:
        with self._lock:
            data = self._load(thread_id)
            data["working_directory"] = working_directory.strip()
            self._save(thread_id, data)
