from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.case_loader import EvalCase, load_cases
from evaluation.scorers.base import ScoreResult
from evaluation.scorers.exact import score_text_expectations
from evaluation.scorers.json_schema import score_json_shape
from evaluation.scorers.llm_judge import score_with_llm_judge
from evaluation.scorers.rag import score_rag_usage
from evaluation.scorers.tool_trace import score_tool_trace
from evaluation.trace import EvalTrace, extract_tool_trace


@dataclass
class EvalOutcome:
    case: EvalCase
    answer: str
    trace: EvalTrace
    scores: list[ScoreResult]
    error: str = ""

    @property
    def aggregate_score(self) -> float:
        visible = [item.score / item.max_score for item in self.scores if not item.details.get("skipped")]
        if not visible:
            return 1.0 if not self.error else 0.0
        return statistics.fmean(visible)

    @property
    def passed(self) -> bool:
        return not self.error and all(item.passed for item in self.scores)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case.id,
            "category": self.case.category,
            "thread_id": self.case.thread_id,
            "passed": self.passed,
            "aggregate_score": self.aggregate_score,
            "answer": self.answer,
            "error": self.error,
            "scores": [item.to_dict() for item in self.scores],
            "trace": self.trace.to_dict(),
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    case_dir = (root / args.case_dir).resolve()
    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    cases = load_cases(case_dir, suites, include_disabled=args.include_disabled)

    if args.dry_run:
        print(f"Loaded {len(cases)} eval cases from {case_dir}")
        for case in cases:
            print(f"- {case.id} [{case.category}] enabled={case.enabled}")
        return 0

    orchestrator = build_orchestrator(root / args.config)
    run_dir = create_run_dir(root / args.output_dir)
    outcomes = []
    for case in cases:
        outcome = run_case(orchestrator, case, use_llm_judge=not args.no_llm_judge)
        outcomes.append(outcome)
        print_case_result(outcome)

    write_results(run_dir, outcomes, suites=suites)
    print(f"Eval run written to {run_dir}")
    return 0 if all(item.passed for item in outcomes) else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Falco offline evaluations.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--config", default="config.yaml", help="Falco config path relative to root.")
    parser.add_argument("--case-dir", default="evaluation/cases", help="Case directory relative to root.")
    parser.add_argument("--suite", default="smoke", help="Comma-separated suite names without .jsonl.")
    parser.add_argument("--output-dir", default="evaluation/runs", help="Output directory relative to root.")
    parser.add_argument("--dry-run", action="store_true", help="Only load and validate cases; do not call the agent.")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled cases.")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip optional LLM-as-judge scoring.")
    return parser.parse_args(argv)


def build_orchestrator(config_path: Path):
    from harness.agents.secretary.wake import FalcoOrchestrator
    from harness.config.config import FalcoSettings

    settings = FalcoSettings.from_yaml(config_path)
    return FalcoOrchestrator(settings=settings)


def run_case(orchestrator, case: EvalCase, *, use_llm_judge: bool) -> EvalOutcome:
    trace = EvalTrace(case_id=case.id, thread_id=case.thread_id)
    answer = ""
    error = ""
    try:
        result = invoke_with_state(orchestrator, case.input, case.thread_id)
        answer = extract_answer(result)
        messages = list(result.get("messages", [])) if isinstance(result, dict) else []
        for step in case.resume_steps:
            resume_input = str(step.get("input", "")).strip()
            if not resume_input:
                continue
            if bool(step.get("resume", False)):
                result = resume_with_state(orchestrator, resume_input, case.thread_id)
            else:
                result = invoke_with_state(orchestrator, resume_input, case.thread_id)
            answer = extract_answer(result)
            messages = list(result.get("messages", [])) if isinstance(result, dict) else messages
        trace.tool_calls = extract_tool_trace(messages)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        trace.errors.append(error)
    finally:
        trace.finish()

    scores = score_case(case, answer, trace, orchestrator=orchestrator, use_llm_judge=use_llm_judge)
    if error:
        scores.append(ScoreResult(name="runtime_error", score=0.0, passed=False, details={"error": error}))
    return EvalOutcome(case=case, answer=answer, trace=trace, scores=scores, error=error)


def invoke_with_state(orchestrator, user_input: str, thread_id: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    orchestrator.memory_postprocess.flush_thread(thread_id)
    orchestrator._refresh_mcp_runtime_if_needed()
    orchestrator._thread_id_ctx.set(thread_id)
    orchestrator._latest_user_ctx.set(user_input)
    orchestrator._response_preference_ctx.set("concise")
    orchestrator._resume_input_ctx.set("")
    working_directory = orchestrator._restore_thread_working_directory(thread_id)
    orchestrator._working_directory_ctx.set(working_directory)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(12, orchestrator.settings.max_tool_steps * 3 + 8),
    }
    state = {
        "messages": [HumanMessage(content=user_input)],
        "thread_id": thread_id,
        "user_response_preference": "concise",
        "working_directory": working_directory,
    }
    return orchestrator.graph.invoke(state, config=config)


def resume_with_state(orchestrator, user_input: str, thread_id: str) -> dict[str, Any]:
    from langgraph.types import Command

    orchestrator.memory_postprocess.flush_thread(thread_id)
    orchestrator._refresh_mcp_runtime_if_needed()
    orchestrator._thread_id_ctx.set(thread_id)
    orchestrator._latest_user_ctx.set(user_input)
    orchestrator._response_preference_ctx.set("concise")
    orchestrator._resume_input_ctx.set(user_input)
    orchestrator._working_directory_ctx.set(orchestrator._restore_thread_working_directory(thread_id))
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(12, orchestrator.settings.max_tool_steps * 3 + 8),
    }
    return orchestrator.graph.invoke(Command(resume=user_input), config=config)


def extract_answer(result: dict[str, Any]) -> str:
    from harness.agents.tool_calling import sanitize_final_answer_text

    if not isinstance(result, dict):
        return ""
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        payload = getattr(interrupts[0], "value", interrupts[0])
        return json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
    for message in reversed(result.get("messages", [])):
        class_name = message.__class__.__name__
        if class_name == "AIMessage" and str(message.content or "").strip():
            return sanitize_final_answer_text(message.content)
        if class_name == "ToolMessage" and str(message.content or "").strip():
            return str(message.content)
    return ""


def score_case(case: EvalCase, answer: str, trace: EvalTrace, *, orchestrator, use_llm_judge: bool) -> list[ScoreResult]:
    scores = [
        score_text_expectations(case, answer),
        score_json_shape(case, answer),
        score_tool_trace(case, trace),
        score_rag_usage(case, trace, answer),
    ]
    if use_llm_judge:
        scores.append(
            score_with_llm_judge(
                case,
                answer,
                model=orchestrator.settings.model,
                base_url=orchestrator.settings.base_url,
            )
        )
    return scores


def create_run_dir(output_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_results(run_dir: Path, outcomes: list[EvalOutcome], *, suites: list[str]) -> None:
    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.to_json(), ensure_ascii=False) + "\n")

    passed = sum(1 for item in outcomes if item.passed)
    total = len(outcomes)
    mean_score = statistics.fmean([item.aggregate_score for item in outcomes]) if outcomes else 0.0
    lines = [
        "# Falco Eval Summary",
        "",
        f"- Suites: {', '.join(suites)}",
        f"- Cases: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Mean score: {mean_score:.3f}",
        "",
        "## Cases",
        "",
    ]
    for outcome in outcomes:
        status = "PASS" if outcome.passed else "FAIL"
        lines.append(f"- {status} {outcome.case.id}: {outcome.aggregate_score:.3f}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_case_result(outcome: EvalOutcome) -> None:
    status = "PASS" if outcome.passed else "FAIL"
    print(f"{status} {outcome.case.id} score={outcome.aggregate_score:.3f} tools={len(outcome.trace.tool_calls)}")
    if outcome.error:
        print(f"  error: {outcome.error}")


if __name__ == "__main__":
    sys.exit(main())
