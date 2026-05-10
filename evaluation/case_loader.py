from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    category: str = "general"
    thread_id: str = ""
    enabled: bool = True
    resume_steps: tuple[dict[str, Any], ...] = ()
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    max_tool_calls: int | None = None
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    expected_json_schema: dict[str, Any] | None = None
    rubric: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, source: Path, line_number: int) -> "EvalCase":
        case_id = str(data.get("id", "")).strip()
        prompt = str(data.get("input", "")).strip()
        if not case_id:
            raise ValueError(f"{source}:{line_number}: missing required field 'id'")
        if not prompt:
            raise ValueError(f"{source}:{line_number}: missing required field 'input'")

        rubric = data.get("rubric") if isinstance(data.get("rubric"), dict) else {}
        return cls(
            id=case_id,
            input=prompt,
            category=str(data.get("category", "general")).strip() or "general",
            thread_id=str(data.get("thread_id", "")).strip() or f"eval_{case_id}",
            enabled=bool(data.get("enabled", True)),
            resume_steps=tuple(item for item in data.get("resume_steps", ()) if isinstance(item, dict)),
            expected_tools=_str_tuple(data.get("expected_tools", ())),
            forbidden_tools=_str_tuple(data.get("forbidden_tools", ())),
            max_tool_calls=_optional_int(data.get("max_tool_calls")),
            must_include=_str_tuple(data.get("must_include", rubric.get("must_include", ()))),
            must_not_include=_str_tuple(data.get("must_not_include", rubric.get("must_not_include", ()))),
            expected_json_schema=data.get("expected_json_schema") if isinstance(data.get("expected_json_schema"), dict) else None,
            rubric=rubric,
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_cases(case_dir: Path, suites: list[str] | tuple[str, ...], *, include_disabled: bool = False) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for suite in suites:
        path = case_dir / f"{suite}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Eval suite not found: {path}")
        cases.extend(load_case_file(path, include_disabled=include_disabled))
    return cases


def load_case_file(path: Path, *, include_disabled: bool = False) -> list[EvalCase]:
    items: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            case = EvalCase.from_mapping(data, source=path, line_number=line_number)
            if case.enabled or include_disabled:
                items.append(case)
    return items

