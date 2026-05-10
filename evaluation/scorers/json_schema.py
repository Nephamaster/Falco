from __future__ import annotations

import json
from typing import Any

from evaluation.case_loader import EvalCase
from evaluation.scorers.base import ScoreResult


def score_json_shape(case: EvalCase, answer: str) -> ScoreResult:
    schema = case.expected_json_schema
    if not schema:
        return ScoreResult(name="json_shape", score=1.0, details={"skipped": True})
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as exc:
        return ScoreResult(name="json_shape", score=0.0, passed=False, details={"error": str(exc)})

    missing = []
    wrong_type = []
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
    for key in required:
        if key not in payload:
            missing.append(key)
    for key, spec in properties.items():
        if key not in payload or not isinstance(spec, dict):
            continue
        expected_type = spec.get("type")
        if expected_type and not _matches_type(payload[key], str(expected_type)):
            wrong_type.append({"key": key, "expected": expected_type, "actual": type(payload[key]).__name__})
    passed = not missing and not wrong_type
    return ScoreResult(
        name="json_shape",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"missing": missing, "wrong_type": wrong_type},
    )


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True

