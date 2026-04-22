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

            from harness.agents.secretary.wake import FalcoOrchestrator
            from harness.config.config import FalcoSettings

            settings = FalcoSettings.from_yaml(root / "config.yaml")
            self._orchestrator = FalcoOrchestrator(settings=settings)
            return self._orchestrator


runtime = FalcoRuntime()
