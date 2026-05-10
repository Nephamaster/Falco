from __future__ import annotations

from evaluation.case_loader import EvalCase
from evaluation.scorers.base import ScoreResult
from evaluation.trace import EvalTrace


def score_rag_usage(case: EvalCase, trace: EvalTrace, answer: str) -> ScoreResult:
    if case.category != "rag" and not case.rubric.get("requires_rag"):
        return ScoreResult(name="rag_usage", score=1.0, details={"skipped": True})
    used_rag = any(item.name in {"rag_search", "use_skill"} and "rag" in str(item.args).lower() for item in trace.tool_calls)
    answer_has_source_signal = any(marker in answer.lower() for marker in ("source", "doc", "chunk", "knowledge"))
    score = 0.0
    if used_rag:
        score += 0.7
    if answer_has_source_signal:
        score += 0.3
    return ScoreResult(
        name="rag_usage",
        score=score,
        passed=used_rag,
        details={"used_rag": used_rag, "answer_has_source_signal": answer_has_source_signal},
    )

