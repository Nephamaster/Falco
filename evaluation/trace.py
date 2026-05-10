from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolTrace:
    name: str
    tool_call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    output_preview: str = ""
    success: bool = True


@dataclass
class EvalTrace:
    case_id: str
    thread_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    tool_calls: list[ToolTrace] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def latency_ms(self) -> int:
        end = self.finished_at if self.finished_at is not None else time.time()
        return int((end - self.started_at) * 1000)

    def finish(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "thread_id": self.thread_id,
            "latency_ms": self.latency_ms,
            "tool_call_count": len(self.tool_calls),
            "tool_calls": [
                {
                    "name": item.name,
                    "tool_call_id": item.tool_call_id,
                    "args": item.args,
                    "output_preview": item.output_preview,
                    "success": item.success,
                }
                for item in self.tool_calls
            ],
            "errors": list(self.errors),
        }


def extract_tool_trace(messages: list[Any]) -> list[ToolTrace]:
    calls_by_id: dict[str, ToolTrace] = {}
    ordered: list[ToolTrace] = []
    for message in messages:
        class_name = message.__class__.__name__
        if class_name == "AIMessage":
            for tool_call in message.tool_calls or []:
                call_id = str(tool_call.get("id", ""))
                item = ToolTrace(
                    name=str(tool_call.get("name", "")),
                    tool_call_id=call_id,
                    args=tool_call.get("args", {}) if isinstance(tool_call.get("args"), dict) else {},
                )
                calls_by_id[call_id] = item
                ordered.append(item)
        elif class_name == "ToolMessage":
            call_id = str(getattr(message, "tool_call_id", "") or "")
            item = calls_by_id.get(call_id)
            if item is None:
                item = ToolTrace(name=str(getattr(message, "name", "") or ""), tool_call_id=call_id)
                ordered.append(item)
            output = str(message.content or "")
            item.output_preview = output[:500]
            item.success = not _looks_like_tool_failure(output)
    return ordered


def _looks_like_tool_failure(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "tool execution failed",
            "traceback",
            "not found",
            "refusing to",
            "outside allowed",
            "unsupported",
        )
    )
