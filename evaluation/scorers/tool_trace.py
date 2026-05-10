from __future__ import annotations

from evaluation.case_loader import EvalCase
from evaluation.scorers.base import ScoreResult
from evaluation.trace import EvalTrace


def score_tool_trace(case: EvalCase, trace: EvalTrace) -> ScoreResult:
    names = [item.name for item in trace.tool_calls]
    missing = [name for name in case.expected_tools if name not in names]
    forbidden = [name for name in case.forbidden_tools if name in names]
    over_budget = case.max_tool_calls is not None and len(names) > case.max_tool_calls

    checks = len(case.expected_tools) + len(case.forbidden_tools) + (1 if case.max_tool_calls is not None else 0)
    if checks == 0:
        return ScoreResult(name="tool_trace", score=1.0, details={"skipped": True, "tool_calls": names})

    failures = len(missing) + len(forbidden) + (1 if over_budget else 0)
    score = max(0.0, (checks - failures) / checks)
    return ScoreResult(
        name="tool_trace",
        score=score,
        passed=failures == 0,
        details={
            "tool_calls": names,
            "missing_expected": missing,
            "forbidden_present": forbidden,
            "max_tool_calls": case.max_tool_calls,
            "actual_tool_calls": len(names),
            "over_budget": over_budget,
        },
    )

