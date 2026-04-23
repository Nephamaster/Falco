from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class _QueuedMemoryTask:
    thread_id: str
    fn: Callable[[], None]


class MemoryPostprocessQueue:
    """Run memory maintenance tasks in a single background worker to preserve ordering."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue: deque[_QueuedMemoryTask] = deque()
        self._pending_by_thread: dict[str, int] = {}
        self._worker = threading.Thread(target=self._run, name="falco-memory-postprocess", daemon=True)
        self._worker.start()

    def enqueue(self, *, thread_id: str, fn: Callable[[], None]) -> None:
        with self._condition:
            self._queue.append(_QueuedMemoryTask(thread_id=thread_id, fn=fn))
            self._pending_by_thread[thread_id] = self._pending_by_thread.get(thread_id, 0) + 1
            self._condition.notify_all()

    def flush_thread(self, thread_id: str) -> None:
        with self._condition:
            while self._pending_by_thread.get(thread_id, 0) > 0:
                self._condition.wait(timeout=0.1)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                task = self._queue.popleft()
            try:
                task.fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] async memory postprocess failed for thread={task.thread_id}: {exc}")
            finally:
                with self._condition:
                    remaining = self._pending_by_thread.get(task.thread_id, 0) - 1
                    if remaining > 0:
                        self._pending_by_thread[task.thread_id] = remaining
                    else:
                        self._pending_by_thread.pop(task.thread_id, None)
                    self._condition.notify_all()
