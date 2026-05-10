from __future__ import annotations

from evaluation.case_loader import EvalCase
from evaluation.scorers.base import ScoreResult


def score_text_expectations(case: EvalCase, answer: str) -> ScoreResult:
    lowered = answer.lower()
    missing = [item for item in case.must_include if item.lower() not in lowered]
    forbidden = [item for item in case.must_not_include if item.lower() in lowered]
    total_checks = len(case.must_include) + len(case.must_not_include)
    if total_checks == 0:
        return ScoreResult(name="text_expectations", score=1.0, details={"skipped": True})
    passed_checks = total_checks - len(missing) - len(forbidden)
    score = max(0.0, passed_checks / total_checks)
    return ScoreResult(
        name="text_expectations",
        score=score,
        passed=not missing and not forbidden,
        details={"missing": missing, "forbidden_present": forbidden},
    )

