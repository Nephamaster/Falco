from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_thread_id(thread_id: str) -> str:
    safe = "".join(ch for ch in str(thread_id or "default") if ch.isalnum() or ch in ("-", "_"))
    return safe or "default"


@dataclass
class SubAgentTaskManager:
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def _subagents_root(self, runtime_dir: Path) -> Path:
        return (runtime_dir / "subagents").resolve()

    def _worker_root(self, runtime_dir: Path, worker_id: str) -> Path:
        return (self._subagents_root(runtime_dir) / worker_id).resolve()

    def _status_path(self, runtime_dir: Path, worker_id: str) -> Path:
        return self._worker_root(runtime_dir, worker_id) / "status.json"

    def _task_path(self, runtime_dir: Path, worker_id: str) -> Path:
        return self._worker_root(runtime_dir, worker_id) / "task.md"

    def _result_path(self, runtime_dir: Path, worker_id: str) -> Path:
        return self._worker_root(runtime_dir, worker_id) / "result.md"

    def _artifacts_dir(self, runtime_dir: Path, worker_id: str) -> Path:
        return self._worker_root(runtime_dir, worker_id) / "artifacts"

    def _load_status(self, runtime_dir: Path, worker_id: str) -> dict[str, Any] | None:
        path = self._status_path(runtime_dir, worker_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_status(self, runtime_dir: Path, worker_id: str, payload: dict[str, Any]) -> None:
        path = self._status_path(runtime_dir, worker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = _utc_now()
        tmp_path = path.with_name(f"{path.name}.{_utc_now().replace(':', '').replace('.', '')}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _next_worker_id(self, runtime_dir: Path) -> str:
        root = self._subagents_root(runtime_dir)
        root.mkdir(parents=True, exist_ok=True)
        highest = 0
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("worker_"):
                continue
            suffix = name[len("worker_") :]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return f"worker_{highest + 1:03d}"

    def create_task(
        self,
        *,
        thread_id: str,
        runtime_dir: Path,
        task: str,
        context: str = "",
        expected_output: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            worker_id = self._next_worker_id(runtime_dir)
            worker_root = self._worker_root(runtime_dir, worker_id)
            task_path = self._task_path(runtime_dir, worker_id)
            result_path = self._result_path(runtime_dir, worker_id)
            artifacts_dir = self._artifacts_dir(runtime_dir, worker_id)
            worker_root.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            task_lines = [
                f"# Subagent Task: {worker_id}",
                "",
                "## Required task",
                task.strip(),
            ]
            if context.strip():
                task_lines.extend(["", "## Context", context.strip()])
            if expected_output.strip():
                task_lines.extend(["", "## Expected output", expected_output.strip()])
            task_lines.extend(
                [
                    "",
                    "## Output contract",
                    f"- Write the final subagent result to `{result_path}`.",
                    f"- Use `{artifacts_dir}` for optional intermediate artifacts.",
                    "- Do not rely on chat text as the final handoff.",
                ]
            )
            task_path.write_text("\n".join(task_lines).strip() + "\n", encoding="utf-8")

            record = {
                "worker_id": worker_id,
                "thread_id": _safe_thread_id(thread_id),
                "status": "pending",
                "created_at": _utc_now(),
                "task": task.strip(),
                "context": context.strip(),
                "expected_output": expected_output.strip(),
                "worker_root": str(worker_root),
                "task_path": str(task_path),
                "result_path": str(result_path),
                "artifacts_dir": str(artifacts_dir),
            }
            self._save_status(runtime_dir, worker_id, record)
            return record

    def list_tasks(self, *, runtime_dir: Path, status: str = "") -> list[dict[str, Any]]:
        with self._lock:
            root = self._subagents_root(runtime_dir)
            if not root.exists():
                return []
            items: list[dict[str, Any]] = []
            for child in sorted(root.iterdir(), key=lambda path: path.name):
                if not child.is_dir():
                    continue
                record = self._load_status(runtime_dir, child.name)
                if not record:
                    continue
                if status and record.get("status") != status:
                    continue
                items.append(record)
            return items

    def count_active_tasks(self, *, runtime_dir: Path) -> int:
        return sum(1 for item in self.list_tasks(runtime_dir=runtime_dir) if item.get("status") in {"pending", "running"})

    def get_task(self, *, runtime_dir: Path, worker_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._load_status(runtime_dir, worker_id)

    def mark_running(self, *, runtime_dir: Path, worker_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._load_status(runtime_dir, worker_id)
            if record is None:
                raise FileNotFoundError(f"Unknown subagent task: {worker_id}")
            record["status"] = "running"
            record["started_at"] = _utc_now()
            self._save_status(runtime_dir, worker_id, record)
            return record

    def mark_completed(self, *, runtime_dir: Path, worker_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._load_status(runtime_dir, worker_id)
            if record is None:
                raise FileNotFoundError(f"Unknown subagent task: {worker_id}")
            record["status"] = "completed"
            record["completed_at"] = _utc_now()
            self._save_status(runtime_dir, worker_id, record)
            return record

    def mark_failed(self, *, runtime_dir: Path, worker_id: str, reason: str) -> dict[str, Any]:
        with self._lock:
            record = self._load_status(runtime_dir, worker_id)
            if record is None:
                raise FileNotFoundError(f"Unknown subagent task: {worker_id}")
            record["status"] = "failed"
            record["failed_at"] = _utc_now()
            record["error"] = reason.strip()
            self._save_status(runtime_dir, worker_id, record)
            return record

    def read_result(self, *, runtime_dir: Path, worker_id: str) -> str:
        record = self.get_task(runtime_dir=runtime_dir, worker_id=worker_id)
        if record is None:
            raise FileNotFoundError(f"Unknown subagent task: {worker_id}")
        result_path = Path(record["result_path"])
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found for subagent {worker_id}: {result_path}")
        return result_path.read_text(encoding="utf-8")
