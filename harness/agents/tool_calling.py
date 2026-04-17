from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage


def coerce_json_tool_call(message: AIMessage, valid_tool_names: set[str]) -> AIMessage:
    """Convert plain JSON tool-call text into LangChain tool_calls when needed."""
    if message.tool_calls:
        return message

    parsed = _parse_tool_call_payload(message.content)
    if parsed is None:
        return message

    name = str(parsed.get("name") or parsed.get("tool") or parsed.get("tool_name") or "").strip()
    if name not in valid_tool_names:
        return message

    args = parsed.get("args")
    if args is None:
        args = parsed.get("arguments", {})
    if not isinstance(args, dict):
        return message

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"manual_tool_{uuid.uuid4().hex[:10]}",
            }
        ],
        additional_kwargs=message.additional_kwargs,
        response_metadata=message.response_metadata,
    )


def _parse_tool_call_payload(content: Any) -> dict[str, Any] | None:
    text = _extract_text(content).strip()
    if not text:
        return None

    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        normalized = _normalize_tool_call_data(data)
        if normalized is not None:
            return normalized
    return None


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)
    return str(content)


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(item.strip() for item in fenced)

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        candidates.append(text[first_obj : last_obj + 1])

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        candidates.append(text[first_arr : last_arr + 1])
    return candidates


def _normalize_tool_call_data(data: Any) -> dict[str, Any] | None:
    if isinstance(data, list):
        if not data:
            return None
        return _normalize_tool_call_data(data[0])

    if not isinstance(data, dict):
        return None

    if "tool_calls" in data and isinstance(data["tool_calls"], list) and data["tool_calls"]:
        return _normalize_tool_call_data(data["tool_calls"][0])
    if "function_call" in data and isinstance(data["function_call"], dict):
        return _normalize_tool_call_data(data["function_call"])
    if "function" in data and isinstance(data["function"], dict):
        function = data["function"]
        return {
            "name": function.get("name"),
            "args": _coerce_args(function.get("arguments", {})),
        }

    name = data.get("name") or data.get("tool") or data.get("tool_name")
    args = data.get("args", data.get("arguments", {}))
    if name:
        return {"name": name, "args": _coerce_args(args)}
    return None


def _coerce_args(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args
