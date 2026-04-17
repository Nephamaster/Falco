"""Falco personal assistant MVP."""

__all__ = ["FalcoOrchestrator"]


def __getattr__(name: str):
    if name == "FalcoOrchestrator":
        from harness.agents.secretary.wake import FalcoOrchestrator

        return FalcoOrchestrator
    raise AttributeError(name)
