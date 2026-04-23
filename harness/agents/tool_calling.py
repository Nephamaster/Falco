from __future__ import annotations

import json
import re
import uuid
from html import unescape
from typing import Any

from langchain_core.messages import AIMessage


def coerce_json_tool_call(message: AIMessage, valid_tool_names: set[str]) -> AIMessage:
    """Convert provider-specific plain-text tool-call payloads into LangChain tool_calls when needed."""
    cleaned_content = _sanitize_assistant_text(message.content)
    if message.tool_calls:
        normalized_tool_calls = []
        changed = False
        for tool_call in message.tool_calls:
            normalized = dict(tool_call)
            args = _normalize_tool_args(normalized.get("name", ""), normalized.get("args"))
            if args != normalized.get("args"):
                normalized["args"] = args
                changed = True
            normalized_tool_calls.append(normalized)
        if not changed:
            if cleaned_content == message.content:
                return message
            return AIMessage(
                content=cleaned_content,
                tool_calls=normalized_tool_calls,
                additional_kwargs={key: value for key, value in (message.additional_kwargs or {}).items() if key not in {"tool_calls", "function_call"}},
                response_metadata=message.response_metadata,
            )
        additional_kwargs = dict(message.additional_kwargs or {})
        additional_kwargs.pop("tool_calls", None)
        additional_kwargs.pop("function_call", None)
        return AIMessage(
            content=cleaned_content,
            tool_calls=normalized_tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata=message.response_metadata,
        )

    parsed = _parse_tool_call_payload(cleaned_content)
    if parsed is None:
        if cleaned_content == message.content:
            return message
        return AIMessage(
            content=cleaned_content,
            additional_kwargs={key: value for key, value in (message.additional_kwargs or {}).items() if key not in {"tool_calls", "function_call"}},
            response_metadata=message.response_metadata,
        )

    name = str(parsed.get("name") or parsed.get("tool") or parsed.get("tool_name") or "").strip()
    if name not in valid_tool_names:
        if cleaned_content == message.content:
            return message
        return AIMessage(
            content=cleaned_content,
            additional_kwargs={key: value for key, value in (message.additional_kwargs or {}).items() if key not in {"tool_calls", "function_call"}},
            response_metadata=message.response_metadata,
        )

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
        additional_kwargs={key: value for key, value in (message.additional_kwargs or {}).items() if key not in {"tool_calls", "function_call"}},
        response_metadata=message.response_metadata,
    )


def _parse_tool_call_payload(content: Any) -> dict[str, Any] | None:
    text = _sanitize_assistant_text(_extract_text(content))
    if not text:
        return None

    minimax = _parse_minimax_tool_call(text)
    if minimax is not None:
        return minimax

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


def _sanitize_assistant_text(content: Any) -> str:
    text = _extract_text(content)
    if not text:
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking\b[^>]*>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<|thought|>", "").replace("<|/thought|>", "")
    return text.strip()


def sanitize_final_answer_text(content: Any) -> str:
    text = _sanitize_assistant_text(content)
    text = re.sub(
        r"<minimax:tool_call\b[^>]*>.*?</minimax:tool_call>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


def _parse_minimax_tool_call(text: str) -> dict[str, Any] | None:
    block_match = re.search(
        r"<minimax:tool_call\b[^>]*>(.*?)</minimax:tool_call>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return None
    body = block_match.group(1)
    invoke_match = re.search(r"<invoke\b[^>]*name=\"([^\"]+)\"[^>]*>(.*?)</invoke>", body, flags=re.DOTALL | re.IGNORECASE)
    if not invoke_match:
        return None
    name = unescape(invoke_match.group(1)).strip()
    raw_args = invoke_match.group(2)
    args: dict[str, Any] = {}
    for param_name, param_value in re.findall(
        r"<parameter\b[^>]*name=\"([^\"]+)\"[^>]*>(.*?)</parameter>",
        raw_args,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        args[unescape(param_name).strip()] = unescape(param_value).strip()
    if not name:
        return None
    return {"name": name, "args": args}


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


def _normalize_tool_args(name: str, args: Any) -> Any:
    if isinstance(args, dict):
        if name == "use_skill" and "v__args" in args and isinstance(args["v__args"], list):
            normalized = _normalize_tool_args(name, args["v__args"])
            if isinstance(normalized, dict):
                for key, value in args.items():
                    if key.startswith("v__"):
                        continue
                    normalized.setdefault(key, value)
                return normalized
        return {key: value for key, value in args.items() if not str(key).startswith("v__")}
    if name == "use_skill" and isinstance(args, list):
        normalized: dict[str, Any] = {}
        if len(args) >= 1:
            normalized["skill_name"] = args[0]
        if len(args) >= 2:
            normalized["action"] = args[1]
        if len(args) >= 3:
            third = args[2]
            normalized["args"] = third if isinstance(third, dict) else {}
        if len(args) >= 4:
            normalized["approved_request_id"] = args[3]
        return normalized
    if isinstance(args, list):
        return {"args": args}
    return args
