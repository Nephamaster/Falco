from __future__ import annotations

import json
import os

from evaluation.case_loader import EvalCase
from evaluation.scorers.base import ScoreResult


def score_with_llm_judge(case: EvalCase, answer: str, *, model: str = "gpt-4o-mini", base_url: str | None = None) -> ScoreResult:
    if not case.rubric.get("judge_dimensions"):
        return ScoreResult(name="llm_judge", score=1.0, details={"skipped": True})
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        return ScoreResult(name="llm_judge", score=1.0, details={"skipped": True, "reason": "LLM_API_KEY is not set"})

    from langchain_openai import ChatOpenAI

    prompt = {
        "case_id": case.id,
        "input": case.input,
        "answer": answer,
        "rubric": case.rubric,
        "instruction": "Return JSON only: {\"score\": 0.0-1.0, \"passed\": boolean, \"reason\": string}.",
    }
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
    response = llm.invoke(json.dumps(prompt, ensure_ascii=False))
    text = str(response.content or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ScoreResult(name="llm_judge", score=0.0, passed=False, details={"error": "judge returned non-json", "raw": text[:500]})
    score = float(payload.get("score", 0.0))
    return ScoreResult(
        name="llm_judge",
        score=max(0.0, min(1.0, score)),
        passed=bool(payload.get("passed", score >= 0.7)),
        details={"reason": str(payload.get("reason", ""))},
    )
