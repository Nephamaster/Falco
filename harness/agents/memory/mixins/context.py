from __future__ import annotations

from typing import Any

from harness.agents.memory.models import DailyLogRecordDecision
from harness.agents.memory.runtime import today_str, utc_now


class MemoryContextMixin:
    def build_context_block(
        self,
        thread_id: str,
        max_items: int = 12,
        *,
        recent_rounds: int | None = None,
        key_rounds: int | None = None,
        query_hint: str = "",
        max_tokens: int = 9000,
    ) -> str:
        memory = self.load(thread_id)
        return self._build_context_block_from_memory(
            memory=memory,
            max_items=max_items,
            recent_rounds=recent_rounds,
            key_rounds=key_rounds,
            query_hint=query_hint,
            max_tokens=max_tokens,
        )

    def _build_context_block_from_memory(
        self,
        *,
        memory: dict[str, Any],
        max_items: int,
        recent_rounds: int | None,
        key_rounds: int | None,
        query_hint: str,
        max_tokens: int,
    ) -> str:
        facts = memory.get("facts", [])[-max_items:]
        turns = memory.get("turns", [])
        summary = memory.get("global_summary", "").strip()

        recent_n = recent_rounds if recent_rounds is not None else self.recent_rounds
        key_n = key_rounds if key_rounds is not None else self.key_rounds
        recent = turns[-recent_n:] if recent_n > 0 else []
        selected_key = self._select_key_turns(
            turns=turns,
            recent=recent,
            query_hint=query_hint,
            key_limit=key_n,
        )

        long_term = self._retrieve_long_term_context(
            query_hint=query_hint,
            max_tokens=max(1500, int(max_tokens * 0.35)),
        )

        sections: list[str] = []
        if summary:
            sections.append("Global conversation summary:\n" + summary)
        if facts:
            fact_lines = [f"- {item['note']}" for item in facts]
            sections.append("Known memory facts:\n" + "\n".join(fact_lines))
        if selected_key:
            key_lines = []
            for turn in selected_key:
                key_lines.append(
                    f"- Turn {turn['id']} [importance={turn.get('importance', 0)}]\n"
                    f"  user: {self._truncate(turn.get('user', ''), 220)}\n"
                    f"  assistant: {self._truncate(turn.get('assistant', ''), 220)}"
                )
            sections.append("Key historical turns:\n" + "\n".join(key_lines))
        if recent:
            recent_lines = []
            for turn in recent:
                recent_lines.append(
                    f"- Turn {turn['id']}:\n"
                    f"  user: {self._truncate(turn.get('user', ''), 320)}\n"
                    f"  assistant: {self._truncate(turn.get('assistant', ''), 320)}"
                )
            sections.append("Recent conversation turns:\n" + "\n".join(recent_lines))
        if long_term:
            sections.append(long_term)

        rendered = self._render_with_budget(sections, max_tokens=max_tokens)
        if not rendered:
            return ""
        return "<memory>\n" + rendered + "\n</memory>\n"

    def maybe_run_silent_turn_compaction(
        self,
        *,
        thread_id: str,
        llm,
        context_soft_limit_tokens: int,
        context_max_tokens: int,
        silent_turn_cooldown_rounds: int,
        query_hint: str = "",
        runtime_context_tokens: int | None = None,
    ) -> None:
        with self._lock:
            memory = self.load(thread_id)
            turns = memory.get("turns", [])
            if not turns:
                return

            latest_turn_id = int(turns[-1].get("id", 0))
            last_silent = int(memory.get("last_silent_turn_id", 0))
            pending_evicted = [item for item in memory.get("pending_evicted_turns", []) if isinstance(item, dict)]
            cooldown_ready = latest_turn_id - last_silent >= max(1, silent_turn_cooldown_rounds)

            snapshot = self._build_context_block_from_memory(
                memory=memory,
                max_items=self.max_facts,
                recent_rounds=self.recent_rounds,
                key_rounds=self.key_rounds,
                query_hint=query_hint,
                max_tokens=context_max_tokens * 2,
            )
            near_window_limit = len(turns) >= max(1, self.max_rounds - max(4, self.key_rounds * 2))
            snapshot_tokens = self._count_tokens(snapshot)
            context_pressure = snapshot_tokens >= context_soft_limit_tokens or (
                runtime_context_tokens is not None and runtime_context_tokens >= context_soft_limit_tokens
            )
            overflow_pressure = bool(pending_evicted) or near_window_limit
            if not context_pressure and not overflow_pressure:
                return
            if not cooldown_ready and not pending_evicted:
                return

            critical_turns = self._select_silent_critical_turns(
                active_turns=turns,
                pending_evicted_turns=pending_evicted,
                query_hint=query_hint,
            )

            decision = self._silent_turn_decision(
                llm=llm,
                summary=memory.get("global_summary", ""),
                context_snapshot=snapshot,
                latest_turn=turns[-1],
                critical_turns=critical_turns,
            )
            if decision.compressed_summary.strip():
                memory["global_summary"] = self._truncate(decision.compressed_summary.strip(), 1800)
            memory["last_silent_turn_id"] = latest_turn_id
            memory["pending_evicted_turns"] = []
            self.save(thread_id, memory)

            fallback_daily_note = self._build_silent_daily_note_from_turns(critical_turns)
            daily_note = decision.daily_note.strip() or fallback_daily_note
            should_write_daily = decision.write_daily or bool(daily_note and critical_turns)
            if should_write_daily and daily_note:
                self._append_daily_record(
                    day=today_str(),
                    record=self._build_daily_record(
                        thread_id=thread_id,
                        source="silent_maintenance",
                        importance=max(self.importance_threshold, int(turns[-1].get("importance", 7))),
                        decision=DailyLogRecordDecision(
                            should_write=True,
                            summary=daily_note,
                            category="memory_maintenance",
                            confidence=0.8,
                            tags=["silent-turn", "compaction"],
                        ),
                        ts=utc_now(),
                        turn_id=latest_turn_id,
                    ),
                )
            if decision.write_evergreen and decision.evergreen_note.strip():
                self._append_evergreen_entry(
                    note=self._truncate(decision.evergreen_note.strip(), 320),
                    importance=max(self.importance_threshold, int(turns[-1].get("importance", 7))),
                    source="silent",
                    thread_id=thread_id,
                )

    def _select_key_turns(
        self,
        *,
        turns: list[dict[str, Any]],
        recent: list[dict[str, Any]],
        query_hint: str,
        key_limit: int,
    ) -> list[dict[str, Any]]:
        if key_limit <= 0:
            return []
        recent_ids = {int(item.get("id", 0)) for item in recent}
        key_candidates = [
            turn
            for turn in turns
            if int(turn.get("id", 0)) not in recent_ids and (
                bool(turn.get("is_key")) or int(turn.get("importance", 0)) >= self.importance_threshold
            )
        ]
        if len(key_candidates) < key_limit:
            remainder = [turn for turn in turns if int(turn.get("id", 0)) not in recent_ids and turn not in key_candidates]
            remainder = sorted(
                remainder,
                key=lambda item: (
                    self._query_relevance(f"{item.get('user', '')}\n{item.get('assistant', '')}", query_hint),
                    int(item.get("importance", 0)),
                    int(item.get("id", 0)),
                ),
                reverse=True,
            )
            key_candidates.extend(remainder[: key_limit - len(key_candidates)])

        key_candidates = sorted(
            key_candidates,
            key=lambda item: (
                self._query_relevance(f"{item.get('user', '')}\n{item.get('assistant', '')}", query_hint),
                int(item.get("importance", 0)),
                int(item.get("id", 0)),
            ),
            reverse=True,
        )
        return sorted(key_candidates[:key_limit], key=lambda item: int(item.get("id", 0)))

    def _serialize_turn_for_retention(self, turn: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(turn.get("id", 0)),
            "ts": str(turn.get("ts", utc_now())),
            "user": self._truncate(str(turn.get("user", "")), 600),
            "assistant": self._truncate(str(turn.get("assistant", "")), 600),
            "importance": int(turn.get("importance", 0)),
            "importance_reason": self._truncate(str(turn.get("importance_reason", "")), 120),
            "is_key": bool(turn.get("is_key", False)),
        }

    def _select_silent_critical_turns(
        self,
        *,
        active_turns: list[dict[str, Any]],
        pending_evicted_turns: list[dict[str, Any]],
        query_hint: str,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if pending_evicted_turns:
            candidates.extend(
                self._serialize_turn_for_retention(item)
                for item in pending_evicted_turns
                if bool(item.get("is_key")) or int(item.get("importance", 0)) >= self.importance_threshold
            )
        if active_turns:
            at_risk = active_turns[: max(6, self.key_rounds * 2)]
            candidates.extend(
                self._serialize_turn_for_retention(item)
                for item in at_risk
                if bool(item.get("is_key")) or int(item.get("importance", 0)) >= self.importance_threshold
            )
        deduped: dict[int, dict[str, Any]] = {}
        for item in candidates:
            tid = int(item.get("id", 0))
            if tid <= 0:
                continue
            if tid not in deduped or int(item.get("importance", 0)) > int(deduped[tid].get("importance", 0)):
                deduped[tid] = item
        ranked = sorted(
            deduped.values(),
            key=lambda item: (
                self._query_relevance(f"{item.get('user', '')}\n{item.get('assistant', '')}", query_hint),
                int(item.get("importance", 0)),
                int(item.get("id", 0)),
            ),
            reverse=True,
        )
        return ranked[: max(6, self.key_rounds * 3)]

    def _format_critical_turns_for_prompt(self, turns: list[dict[str, Any]]) -> str:
        if not turns:
            return ""
        lines: list[str] = []
        for turn in turns:
            lines.append(
                f"- Turn {turn.get('id', 0)} [importance={turn.get('importance', 0)}]: "
                f"user={self._truncate(str(turn.get('user', '')), 180)} | "
                f"assistant={self._truncate(str(turn.get('assistant', '')), 180)}"
            )
        return "\n".join(lines)

    def _build_silent_daily_note_from_turns(self, turns: list[dict[str, Any]]) -> str:
        if not turns:
            return ""
        lines = [
            f"turn {int(item.get('id', 0))} (importance={int(item.get('importance', 0))}): "
            f"{self._truncate(str(item.get('user', '')), 110)}"
            for item in turns[:4]
        ]
        return self._truncate(
            "Silent retention of at-risk key turns: " + " ; ".join(lines),
            500,
        )
