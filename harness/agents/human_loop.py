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
class HumanLoopManager:
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
            return {"thread_id": thread_id, "requests": [], "updated_at": _utc_now()}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, thread_id: str, payload: dict[str, Any]) -> None:
        path = self._path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = _utc_now()
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def create_clarification(
        self,
        *,
        thread_id: str,
        question: str,
        clarification_type: str = "missing_info",
        context: str = "",
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        item = {
            "id": f"hitl_{uuid.uuid4().hex[:10]}",
            "type": "clarification",
            "status": "pending",
            "created_at": _utc_now(),
            "question": question.strip(),
            "clarification_type": clarification_type.strip() or "missing_info",
            "context": context.strip(),
            "options": [item.strip() for item in (options or []) if item.strip()][:6],
        }
        with self._lock:
            data = self._load(thread_id)
            data.setdefault("requests", []).append(item)
            self._save(thread_id, data)
        return item

    def create_approval(
        self,
        *,
        thread_id: str,
        action: str,
        payload: dict[str, Any],
        rationale: str,
        question: str = "",
    ) -> dict[str, Any]:
        item = {
            "id": f"hitl_{uuid.uuid4().hex[:10]}",
            "type": "approval",
            "status": "pending",
            "created_at": _utc_now(),
            "action": action,
            "payload": payload,
            "question": question.strip(),
            "rationale": rationale.strip(),
        }
        with self._lock:
            data = self._load(thread_id)
            data.setdefault("requests", []).append(item)
            self._save(thread_id, data)
        return item

    def list_pending(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load(thread_id)
        return [item for item in data.get("requests", []) if item.get("status") == "pending"]

    def get_pending(self, thread_id: str, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._load(thread_id)
        for item in data.get("requests", []):
            if item.get("id") == request_id and item.get("status") == "pending":
                return item
        return None

    def mark_completed(self, thread_id: str, request_id: str, result: str) -> None:
        self._mark(thread_id, request_id, "completed", result=result)

    def mark_denied(self, thread_id: str, request_id: str, reason: str = "") -> None:
        self._mark(thread_id, request_id, "denied", reason=reason)

    def _mark(self, thread_id: str, request_id: str, status: str, **extra: Any) -> None:
        with self._lock:
            data = self._load(thread_id)
            for item in data.get("requests", []):
                if item.get("id") == request_id:
                    item["status"] = status
                    item["resolved_at"] = _utc_now()
                    item.update(extra)
                    break
            self._save(thread_id, data)
