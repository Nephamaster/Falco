from __future__ import annotations

import threading
from pathlib import Path


class FalcoRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orchestrator = None

    def get_orchestrator(self):
        if self._orchestrator is not None:
            return self._orchestrator

        with self._lock:
            if self._orchestrator is not None:
                return self._orchestrator

            root = Path(__file__).resolve().parents[2]
            src_path = root / "src"
            import sys

            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from config import FalcoSettings
            from orchestrator import FalcoOrchestrator

            settings = FalcoSettings.from_env(workspace_root=root)
            self._orchestrator = FalcoOrchestrator(settings=settings)
            return self._orchestrator


runtime = FalcoRuntime()
