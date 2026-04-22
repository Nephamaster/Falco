from __future__ import annotations

import re
from typing import Any

from langchain_openai.chat_models.base import ChatOpenAI

from harness.agents.memory.models import (
    DailyLogRecordDecision,
    EvergreenDiaryDecision,
    ImportanceScore,
    ReflectionDecision,
    SilentTurnDecision,
    SummaryUpdate,
)
from harness.agents.memory.prompt import (
    DAILY_LOG_DECISION_PAYLOAD_TEMPLATE,
    DAILY_LOG_DECISION_PROMPT,
    EVERGREEN_DECISION_PAYLOAD_TEMPLATE,
    EVERGREEN_DECISION_PROMPT,
    GLOBAL_SUMMARY_PROMPT,
    REFLECTION_DECISION_PAYLOAD_TEMPLATE,
    REFLECTION_DECISION_PROMPT,
    SCORE_IMPORTANCE_PROMPT,
    SILENT_COMPRESS_PAYLOAD_TEMPLATE,
    SILENT_COMPRESS_PROMPT,
)


class MemoryInferenceMixin:
    def _score_importance(self, user: str, assistant: str, llm: ChatOpenAI = None) -> ImportanceScore:
        if llm is None:
            return self._heuristic_importance(user, assistant)
        dialogue = f"User:\n{user}\nAssistant:\n{assistant}"
        prompt = SCORE_IMPORTANCE_PROMPT.format(dialogue=dialogue)
        try:
            scorer = llm.with_structured_output(ImportanceScore)
            result = scorer.invoke(prompt)
            score = max(1, min(int(result.score), 10))
            return ImportanceScore(score=score, reason=result.reason.strip())
        except Exception:
            print("[WARNING] fallback to heuristic importance")
            return self._heuristic_importance(user, assistant)

    def _heuristic_importance(self, user: str, assistant: str) -> ImportanceScore:
        text = f"{user}\n{assistant}".lower()
        score = 4
        boosts = [
            (r"\b(todo|next step|roadmap|milestone|deadline)\b", 2),
            (r"\b(prefer|always|never|must|constraint|requirement)\b", 2),
            (r"\b(decision|decide|agreed|final)\b", 2),
            (r"\b(bug|error|failed|incident|regression)\b", 2),
            (r"\b(api|schema|architecture|database|deploy)\b", 1),
            (r"(偏好|喜欢|不喜欢|习惯|要求|必须|约束|决定|结论|截止|里程碑|任务|错误|故障|回归|接口|架构|数据库|部署)", 2),
        ]
        for pattern, delta in boosts:
            if re.search(pattern, text):
                score += delta
        score = max(1, min(score, 10))
        return ImportanceScore(score=score, reason="heuristic")

    def _update_global_summary(self, summary: str, turn: dict[str, Any], llm: ChatOpenAI = None) -> str:
        if llm is None:
            return self._fallback_summary(summary, turn)
        new_turn = (
            f"- User: {turn.get('user', '')}\n-Assistant: {turn.get('assistant', '')}\n"
            f"- Importance score: {turn.get('importance', 0)}"
        )
        prompt = GLOBAL_SUMMARY_PROMPT.format(summary=summary, dialogue=new_turn)
        try:
            updater = llm.with_structured_output(SummaryUpdate)
            result = updater.invoke(prompt)
            return self._truncate((result.summary or "").strip(), 1800)
        except Exception:
            return self._fallback_summary(summary, turn)

    def _fallback_summary(self, summary: str, turn: dict[str, Any]) -> str:
        user = self._truncate(turn.get("user", ""), 220)
        assistant = self._truncate(turn.get("assistant", ""), 220)
        merged = summary.strip()
        appendix = f"- U: {user}\n- A: {assistant}"
        if merged:
            merged = f"{merged}\n{appendix}"
        else:
            merged = appendix
        return self._truncate_tail(merged, 1800)

    def _silent_turn_decision(
        self,
        *,
        llm,
        summary: str,
        context_snapshot: str,
        latest_turn: dict[str, Any],
        critical_turns: list[dict[str, Any]],
    ) -> SilentTurnDecision:
        critical_lines = self._format_critical_turns_for_prompt(critical_turns)
        if llm is None:
            fallback_note = self._build_silent_daily_note_from_turns(critical_turns)
            fallback_summary = self._build_silent_fallback_summary(
                summary=summary,
                latest_turn=latest_turn,
                critical_turns=critical_turns,
            )
            return SilentTurnDecision(
                compressed_summary=fallback_summary,
                write_daily=bool(fallback_note) or int(latest_turn.get("importance", 0)) >= self.importance_threshold,
                daily_note=self._truncate(
                    fallback_note or f"silent: {latest_turn.get('user', '')} | {latest_turn.get('assistant', '')}",
                    420,
                ),
                write_evergreen=False,
                evergreen_note="",
            )
        payload = SILENT_COMPRESS_PAYLOAD_TEMPLATE.format(
            summary=summary or "(empty)",
            latest_user=latest_turn.get("user", ""),
            latest_assistant=latest_turn.get("assistant", ""),
            critical_turns=critical_lines or "(none)",
            context_snapshot=self._truncate(context_snapshot, 5000),
        )
        try:
            runner = llm.with_structured_output(SilentTurnDecision)
            result = runner.invoke(
                [
                    {"role": "system", "content": SILENT_COMPRESS_PROMPT},
                    {"role": "user", "content": payload},
                ]
            )
            result.compressed_summary = self._truncate(result.compressed_summary.strip(), 1800)
            result.daily_note = self._truncate(result.daily_note.strip(), 500)
            result.evergreen_note = self._truncate(result.evergreen_note.strip(), 320)
            return result
        except Exception:
            fallback_summary = self._build_silent_fallback_summary(
                summary=summary,
                latest_turn=latest_turn,
                critical_turns=critical_turns,
            )
            return SilentTurnDecision(
                compressed_summary=fallback_summary,
                write_daily=bool(critical_turns),
                daily_note=self._build_silent_daily_note_from_turns(critical_turns),
                write_evergreen=False,
                evergreen_note="",
            )

    def _build_silent_fallback_summary(
        self,
        *,
        summary: str,
        latest_turn: dict[str, Any],
        critical_turns: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = []
        clean_summary = summary.strip()
        if clean_summary:
            sections.append(clean_summary)

        latest_user = self._truncate(str(latest_turn.get("user", "")), 180)
        latest_assistant = self._truncate(str(latest_turn.get("assistant", "")), 180)
        latest_bits = [part for part in (latest_user, latest_assistant) if part]
        if latest_bits:
            sections.append("Latest turn: " + " | ".join(latest_bits))

        if critical_turns:
            retained_lines = []
            for turn in critical_turns[:4]:
                turn_bits = []
                user_text = self._truncate(str(turn.get("user", "")), 120)
                assistant_text = self._truncate(str(turn.get("assistant", "")), 120)
                if user_text:
                    turn_bits.append(f"user={user_text}")
                if assistant_text:
                    turn_bits.append(f"assistant={assistant_text}")
                if turn_bits:
                    retained_lines.append(
                        f"Turn {int(turn.get('id', 0))} (importance={int(turn.get('importance', 0))}): "
                        + " | ".join(turn_bits)
                    )
            if retained_lines:
                sections.append("Retained key turns:\n" + "\n".join(retained_lines))

        if not sections:
            return ""
        return self._truncate("\n\n".join(sections), 1800)

    def _extract_evergreen_note(self, user: str, assistant: str) -> str:
        text = user.strip()
        if not text:
            return ""
        indicator = re.search(
            r"(喜欢|偏好|习惯|不喜欢|我一般|我通常|长期|风格|I like|I prefer|I usually|I don't like)",
            text,
            re.IGNORECASE,
        )
        if not indicator:
            return ""
        return self._truncate(text, 260)

    def _extract_evergreen_decision(
        self,
        *,
        user: str,
        assistant: str,
        importance: int,
        llm=None,
    ) -> EvergreenDiaryDecision | None:
        if llm is not None:
            payload = EVERGREEN_DECISION_PAYLOAD_TEMPLATE.format(
                importance=importance,
                user=user,
                assistant=assistant,
            )
            try:
                extractor = llm.with_structured_output(EvergreenDiaryDecision)
                decision = extractor.invoke(
                    [
                        {"role": "system", "content": EVERGREEN_DECISION_PROMPT},
                        {"role": "user", "content": payload},
                    ]
                )
                decision.note = self._truncate(decision.note.strip(), 320)
                decision.tags = [self._truncate(item.strip(), 40) for item in decision.tags if item.strip()][:8]
                if decision.should_write and decision.note:
                    return decision
            except Exception:
                pass
        fallback_note = self._extract_evergreen_note(user, assistant)
        if not fallback_note:
            return None
        return EvergreenDiaryDecision(
            should_write=True,
            note=fallback_note,
            confidence=0.55 if importance < self.importance_threshold else 0.65,
            tags=["preference"],
        )

    def _build_reflection_decision(
        self,
        *,
        user: str,
        assistant: str,
        tool_observations: list[str],
        llm,
    ) -> ReflectionDecision | None:
        observations = "\n".join(f"- {self._truncate(item, 500)}" for item in tool_observations[:8])
        payload = REFLECTION_DECISION_PAYLOAD_TEMPLATE.format(
            user=self._truncate(user, 1200),
            assistant=self._truncate(assistant, 1200),
            observations=observations or "(none)",
        )
        try:
            reflector = llm.with_structured_output(ReflectionDecision)
            result = reflector.invoke(
                [
                    {"role": "system", "content": REFLECTION_DECISION_PROMPT},
                    {"role": "user", "content": payload},
                ]
            )
            result.lesson = self._truncate(result.lesson.strip(), 320)
            result.trigger = self._truncate(result.trigger.strip(), 240)
            result.recommendation = self._truncate(result.recommendation.strip(), 260)
            result.tags = [self._truncate(item.strip(), 40) for item in result.tags if item.strip()][:8]
            return result
        except Exception:
            print("fallback to heuristic reflection")
            return self._heuristic_reflection(user=user, assistant=assistant, tool_observations=tool_observations)

    def _heuristic_reflection(
        self,
        *,
        user: str,
        assistant: str,
        tool_observations: list[str],
    ) -> ReflectionDecision | None:
        joined = "\n".join([user, assistant, *tool_observations]).lower()
        if not re.search(r"\b(error|failed|refusing|not found|traceback|exception|path escapes|tool execution failed)\b", joined):
            return None
        return ReflectionDecision(
            should_write=True,
            lesson="When tool output indicates failure, inspect the exact observation and adjust the next action instead of repeating the same call.",
            trigger=self._truncate(" | ".join(tool_observations[:2]), 240),
            recommendation="Summarize the failure, choose a smaller verification step, then retry with corrected inputs.",
            confidence=0.7,
            tags=["tool-use", "error-recovery"],
        )

    def _extract_daily_log_record(
        self,
        *,
        user: str,
        assistant: str,
        importance: int,
        llm=None,
    ) -> DailyLogRecordDecision | None:
        if llm is not None:
            payload = DAILY_LOG_DECISION_PAYLOAD_TEMPLATE.format(
                importance=importance,
                user=user,
                assistant=assistant,
            )
            try:
                extractor = llm.with_structured_output(DailyLogRecordDecision)
                decision = extractor.invoke(
                    [
                        {"role": "system", "content": DAILY_LOG_DECISION_PROMPT},
                        {"role": "user", "content": payload},
                    ]
                )
                decision.summary = self._truncate(decision.summary.strip(), 500)
                decision.facts = [self._truncate(item.strip(), 180) for item in decision.facts if item.strip()][:8]
                decision.decisions = [self._truncate(item.strip(), 180) for item in decision.decisions if item.strip()][:6]
                decision.tasks = [self._truncate(item.strip(), 180) for item in decision.tasks if item.strip()][:6]
                decision.user_preferences = [self._truncate(item.strip(), 180) for item in decision.user_preferences if item.strip()][:6]
                decision.constraints = [self._truncate(item.strip(), 180) for item in decision.constraints if item.strip()][:6]
                decision.artifacts = [self._truncate(item.strip(), 180) for item in decision.artifacts if item.strip()][:8]
                decision.next_actions = [self._truncate(item.strip(), 180) for item in decision.next_actions if item.strip()][:6]
                decision.tags = [self._truncate(item.strip(), 40) for item in decision.tags if item.strip()][:8]
                if decision.should_write and self._daily_decision_has_content(decision):
                    return decision
            except Exception:
                pass
        return self._extract_daily_log_record_heuristic(user=user, assistant=assistant, importance=importance)

    def _daily_decision_has_content(self, decision: DailyLogRecordDecision) -> bool:
        return bool(
            decision.summary.strip()
            or decision.facts
            or decision.decisions
            or decision.tasks
            or decision.user_preferences
            or decision.constraints
            or decision.artifacts
            or decision.next_actions
        )

    def _extract_daily_log_record_heuristic(
        self,
        *,
        user: str,
        assistant: str,
        importance: int,
    ) -> DailyLogRecordDecision | None:
        joined = f"{user}\n{assistant}".strip()
        if not joined:
            return None
        has_fact_signal = bool(
            re.search(
                r"(事实|结论|决定|任务|版本|日期|截止|约束|必须|偏好|要求|deadline|due|version|date|must|constraint|prefer)",
                joined,
                re.IGNORECASE,
            )
        )
        if importance < self.importance_threshold and not has_fact_signal:
            return None
        facts = self._heuristic_extract_facts(joined)
        tasks = self._heuristic_extract_by_signal(
            joined,
            r"\b(todo|task|next step|deadline|due|follow up|implement|fix|deploy|test)\b",
        )
        constraints = self._heuristic_extract_by_signal(
            joined,
            r"\b(must|never|always|constraint|requirement|do not|don't|avoid)\b",
        )
        preferences = self._heuristic_extract_by_signal(
            joined,
            r"\b(prefer|like|usually|style|habit)\b|偏好|喜欢|习惯|风格",
        )
        decisions = self._heuristic_extract_by_signal(
            joined,
            r"\b(decide|decision|agreed|chosen|final)\b|决定|结论|已确认",
        )
        return DailyLogRecordDecision(
            should_write=True,
            summary=self._truncate(joined, 340),
            category=self._infer_daily_category(joined, tasks=tasks, preferences=preferences, constraints=constraints),
            confidence=0.55 if not facts else 0.65,
            facts=facts,
            decisions=decisions,
            tasks=tasks,
            user_preferences=preferences,
            constraints=constraints,
            tags=self._infer_daily_tags(joined),
        )

    def _heuristic_extract_facts(self, text: str) -> list[str]:
        lines = re.split(r"[\n,;，；。]|(?<=[.!?])\s+", text)
        facts: list[str] = []
        for line in lines:
            piece = line.strip()
            if len(piece) < 6:
                continue
            if re.search(
                r"(事实|关键信息|偏好|习惯|要求|任务|版本|日期|截止|deadline|version|date|must|prefer|usually)",
                piece,
                re.IGNORECASE,
            ):
                facts.append(self._truncate(piece, 160))
            if len(facts) >= 6:
                break
        return facts

    def _heuristic_extract_by_signal(self, text: str, pattern: str, limit: int = 4) -> list[str]:
        lines = re.split(r"[\n,;，；。]|(?<=[.!?])\s+", text)
        items: list[str] = []
        for line in lines:
            piece = line.strip()
            if len(piece) < 6:
                continue
            if re.search(pattern, piece, re.IGNORECASE):
                items.append(self._truncate(piece, 180))
            if len(items) >= limit:
                break
        return items

    def _infer_daily_category(
        self,
        text: str,
        *,
        tasks: list[str],
        preferences: list[str],
        constraints: list[str],
    ) -> str:
        low = text.lower()
        if preferences:
            return "preference"
        if constraints:
            return "constraint"
        if tasks or re.search(r"\b(todo|task|deadline|due|implement|fix|deploy|test)\b", low):
            return "task"
        if re.search(r"\b(decision|decide|agreed|final)\b", low):
            return "decision"
        if re.search(r"\b(error|bug|failed|incident|regression)\b", low):
            return "info"
        return "other"

    def _infer_daily_tags(self, text: str) -> list[str]:
        tags: list[str] = []
        signals = [
            ("task", r"\b(todo|task|deadline|due|implement|fix)\b"),
            ("preference", r"\b(prefer|like|usually|style|habit)\b|偏好|喜欢|习惯"),
            ("constraint", r"\b(must|never|always|constraint|requirement)\b"),
            ("bug", r"\b(error|bug|failed|incident|regression)\b"),
            ("architecture", r"\b(api|schema|architecture|database|agent|memory|rag)\b"),
        ]
        for tag, pattern in signals:
            if re.search(pattern, text, re.IGNORECASE):
                tags.append(tag)
        return tags[:6]
