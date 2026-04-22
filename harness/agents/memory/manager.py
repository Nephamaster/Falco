from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from harness.agents.memory.mixins.context import MemoryContextMixin
from harness.agents.memory.mixins.inference import MemoryInferenceMixin
from harness.agents.memory.mixins.store import MemoryStoreMixin
from harness.agents.memory.mixins.utils import MemoryUtilityMixin
from harness.agents.memory.models import (
    EVERGREEN_REFLECTION_MODULE,
)
from harness.agents.memory.runtime import today_str, utc_now


@dataclass
class ConversationMemoryManager(
    MemoryContextMixin,
    MemoryInferenceMixin,
    MemoryStoreMixin,
    MemoryUtilityMixin,
):
    """Facade coordinator for memory persistence, retrieval and maintenance."""

    root: Path
    max_history: int = 60
    max_facts: int = 50
    recent_rounds: int = 6
    key_rounds: int = 4
    importance_threshold: int = 7
    max_rounds: int = 160
    daily_half_life_days: int = 30
    daily_lookback_days: int = 180
    daily_retrieval_items: int = 8
    evergreen_retrieval_items: int = 5
    tokenizer_model: str = "gpt-4o-mini"
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.daily_root.mkdir(parents=True, exist_ok=True)
        if not self.evergreen_path.exists():
            self._write_json_atomic(self.evergreen_path, self._empty_evergreen())

    def add_round(
        self,
        thread_id: str,
        *,
        user: str,
        assistant: str,
        llm=None,
    ) -> None:
        user_text = user.strip()
        assistant_text = assistant.strip()
        if not user_text and not assistant_text:
            return

        with self._lock:
            memory = self.load(thread_id)
            turns = memory.setdefault("turns", [])
            turn_id = int(memory.get("next_turn_id", 1))
            score_info = self._score_importance(user_text, assistant_text, llm=llm)
            turn = {
                "id": turn_id,
                "ts": utc_now(),
                "user": user_text,
                "assistant": assistant_text,
                "importance": score_info.score,
                "importance_reason": score_info.reason,
                "is_key": score_info.score >= self.importance_threshold,
            }
            turns.append(turn)
            if len(turns) > self.max_rounds:
                evicted = turns[: len(turns) - self.max_rounds]
                pending_evicted = memory.setdefault("pending_evicted_turns", [])
                pending_evicted.extend(self._serialize_turn_for_retention(item) for item in evicted)
                memory["pending_evicted_turns"] = pending_evicted[-max(24, self.key_rounds * 10) :]
            memory["turns"] = turns[-self.max_rounds :]
            memory["next_turn_id"] = turn_id + 1

            summary = memory.get("global_summary", "")
            memory["global_summary"] = self._update_global_summary(summary=summary, turn=turn, llm=llm)

            history = memory.setdefault("history", [])
            if user_text:
                history.append({"ts": turn["ts"], "role": "user", "content": user_text})
            if assistant_text:
                history.append({"ts": turn["ts"], "role": "assistant", "content": assistant_text})
            memory["history"] = history[-self.max_history :]

            daily_record = self._extract_daily_log_record(
                user=user_text,
                assistant=assistant_text,
                importance=int(turn["importance"]),
                llm=llm,
            )
            if daily_record is not None:
                self._append_daily_record(
                    day=today_str(),
                    record=self._build_daily_record(
                        thread_id=thread_id,
                        source="dialogue_turn",
                        importance=int(turn["importance"]),
                        decision=daily_record,
                        ts=turn["ts"],
                        turn_id=int(turn["id"]),
                        raw_user=user_text,
                        raw_assistant=assistant_text,
                    ),
                )

            evergreen_decision = self._extract_evergreen_decision(
                user=user_text,
                assistant=assistant_text,
                importance=int(turn["importance"]),
                llm=llm,
            )
            if evergreen_decision is not None and evergreen_decision.should_write and evergreen_decision.note.strip():
                self._append_evergreen_entry(
                    note=evergreen_decision.note.strip(),
                    importance=max(6, int(turn["importance"])),
                    source="round",
                    thread_id=thread_id,
                    confidence=evergreen_decision.confidence,
                    tags=evergreen_decision.tags,
                )

            self.save(thread_id, memory)

    def reflect_on_turn(
        self,
        *,
        thread_id: str,
        user: str,
        assistant: str,
        tool_observations: list[str],
        llm,
    ) -> None:
        if not user.strip() and not assistant.strip():
            return
        decision = self._build_reflection_decision(
            user=user,
            assistant=assistant,
            tool_observations=tool_observations,
            llm=llm,
        )
        if decision is None or not decision.should_write or not decision.lesson.strip():
            return
        if decision.confidence < 0.65:
            return
        with self._lock:
            self._append_evergreen_entry(
                note=self._truncate(decision.lesson.strip(), 320),
                importance=8 if decision.confidence >= 0.8 else 7,
                source="reflexion",
                thread_id=thread_id,
                module=EVERGREEN_REFLECTION_MODULE,
                confidence=decision.confidence,
                tags=[self._truncate(item.strip(), 40) for item in decision.tags if item.strip()][:8],
                trigger=self._truncate(decision.trigger.strip(), 240),
                recommendation=self._truncate(decision.recommendation.strip(), 260),
            )
