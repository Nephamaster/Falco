from __future__ import annotations

import json
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_ts(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


class ImportanceScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = Field(default="")


class SummaryUpdate(BaseModel):
    summary: str = Field(description="Updated compact global summary.")


class SilentTurnDecision(BaseModel):
    compressed_summary: str = Field(default="")
    write_daily: bool = Field(default=False)
    daily_note: str = Field(default="")
    write_evergreen: bool = Field(default=False)
    evergreen_note: str = Field(default="")


class DailyLogRecordDecision(BaseModel):
    should_write: bool = Field(default=False)
    summary: str = Field(default="")
    category: str = Field(default="conversation")
    confidence: float = Field(default=0.7, ge=0, le=1)
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ReflectionDecision(BaseModel):
    should_write: bool = Field(default=False)
    lesson: str = Field(default="")
    trigger: str = Field(default="")
    recommendation: str = Field(default="")
    confidence: float = Field(default=0.7, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


DAILY_LOG_SCHEMA_VERSION = 2
EVERGREEN_SCHEMA_VERSION = 2
EVERGREEN_USER_MODULE = "user"
EVERGREEN_REFLECTION_MODULE = "agent_reflections"


@dataclass
class ConversationMemoryManager:
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
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.daily_root.mkdir(parents=True, exist_ok=True)
        if not self.evergreen_path.exists():
            self._write_json_atomic(self.evergreen_path, self._empty_evergreen())

    @property
    def daily_root(self) -> Path:
        return self.root / "daily"

    @property
    def evergreen_path(self) -> Path:
        return self.root / "evergreen.json"

    def _path(self, thread_id: str) -> Path:
        safe = "".join(ch for ch in thread_id if ch.isalnum() or ch in ("-", "_")) or "default"
        return self.root / f"{safe}.json"

    def load(self, thread_id: str) -> dict:
        path = self._path(thread_id)
        if not path.exists():
            return {
                "thread_id": thread_id,
                "updated_at": _utc_now(),
                "facts": [],
                "history": [],
                "turns": [],
                "global_summary": "",
                "last_silent_turn_id": 0,
            }
        memory = json.loads(path.read_text(encoding="utf-8"))
        memory.setdefault("facts", [])
        memory.setdefault("history", [])
        memory.setdefault("turns", [])
        memory.setdefault("global_summary", "")
        memory.setdefault("last_silent_turn_id", 0)
        if not memory["turns"] and memory["history"]:
            memory["turns"] = self._history_to_turns(memory["history"])
        return memory

    def save(self, thread_id: str, memory: dict) -> None:
        memory["updated_at"] = _utc_now()
        self._write_json_atomic(self._path(thread_id), memory)

    def add_fact(self, thread_id: str, note: str) -> None:
        with self._lock:
            memory = self.load(thread_id)
            facts = memory.setdefault("facts", [])
            facts.append({"ts": _utc_now(), "note": note.strip()})
            memory["facts"] = facts[-self.max_facts :]
            self.save(thread_id, memory)

    def append_turn(self, thread_id: str, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        with self._lock:
            memory = self.load(thread_id)
            history = memory.setdefault("history", [])
            history.append({"ts": _utc_now(), "role": role, "content": text})
            memory["history"] = history[-self.max_history :]
            self.save(thread_id, memory)

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
            turn_id = len(turns) + 1
            score_info = self._score_importance(user_text, assistant_text, llm=llm)
            turn = {
                "id": turn_id,
                "ts": _utc_now(),
                "user": user_text,
                "assistant": assistant_text,
                "importance": score_info.score,
                "importance_reason": score_info.reason,
                "is_key": score_info.score >= self.importance_threshold,
            }
            turns.append(turn)
            memory["turns"] = turns[-self.max_rounds :]

            summary = memory.get("global_summary", "")
            memory["global_summary"] = self._update_global_summary(
                summary=summary,
                turn=turn,
                llm=llm,
            )

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
                    day=_today_str(),
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

            evergreen_note = self._extract_evergreen_note(user_text, assistant_text)
            if evergreen_note:
                self._append_evergreen_entry(
                    note=evergreen_note,
                    importance=max(6, int(turn["importance"])),
                    source="round",
                    thread_id=thread_id,
                )

            self.save(thread_id, memory)

    def maybe_run_silent_turn_compaction(
        self,
        *,
        thread_id: str,
        llm,
        context_soft_limit_chars: int,
        context_max_chars: int,
        silent_turn_cooldown_rounds: int,
        query_hint: str = "",
    ) -> None:
        with self._lock:
            memory = self.load(thread_id)
            turns = memory.get("turns", [])
            if not turns:
                return

            latest_turn_id = int(turns[-1].get("id", 0))
            last_silent = int(memory.get("last_silent_turn_id", 0))
            if latest_turn_id - last_silent < max(1, silent_turn_cooldown_rounds):
                return

            snapshot = self.build_context_block(
                thread_id=thread_id,
                max_items=self.max_facts,
                recent_rounds=self.recent_rounds,
                key_rounds=self.key_rounds,
                query_hint=query_hint,
                max_chars=context_max_chars * 2,
            )
            if len(snapshot) < context_soft_limit_chars:
                return

            decision = self._silent_turn_decision(
                llm=llm,
                summary=memory.get("global_summary", ""),
                context_snapshot=snapshot,
                latest_turn=turns[-1],
            )
            if decision.compressed_summary.strip():
                memory["global_summary"] = decision.compressed_summary.strip()[:1800]
            memory["last_silent_turn_id"] = latest_turn_id
            self.save(thread_id, memory)

            if decision.write_daily and decision.daily_note.strip():
                self._append_daily_record(
                    day=_today_str(),
                    record=self._build_daily_record(
                        thread_id=thread_id,
                        source="silent_maintenance",
                        importance=max(self.importance_threshold, int(turns[-1].get("importance", 7))),
                        decision=DailyLogRecordDecision(
                            should_write=True,
                            summary=decision.daily_note.strip(),
                            category="memory_maintenance",
                            confidence=0.8,
                            tags=["silent-turn", "compaction"],
                        ),
                        ts=_utc_now(),
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

    def build_context_block(
        self,
        thread_id: str,
        max_items: int = 12,
        *,
        recent_rounds: int | None = None,
        key_rounds: int | None = None,
        query_hint: str = "",
        max_chars: int = 9000,
    ) -> str:
        memory = self.load(thread_id)
        facts = memory.get("facts", [])[-max_items:]
        turns = memory.get("turns", [])
        summary = memory.get("global_summary", "").strip()

        recent_n = recent_rounds if recent_rounds is not None else self.recent_rounds
        key_n = key_rounds if key_rounds is not None else self.key_rounds
        recent = turns[-recent_n:] if recent_n > 0 else []
        recent_ids = {item["id"] for item in recent}
        key_candidates = [turn for turn in turns if turn.get("id") not in recent_ids]
        key_candidates = sorted(
            key_candidates,
            key=lambda item: (int(item.get("importance", 0)), int(item.get("id", 0))),
            reverse=True,
        )
        selected_key = sorted(key_candidates[:key_n], key=lambda item: int(item.get("id", 0)))

        long_term = self._retrieve_long_term_context(query_hint=query_hint, max_chars=max(2000, int(max_chars * 0.35)))

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

        rendered = self._render_with_budget(sections, max_chars=max_chars)
        if not rendered:
            return ""
        return "<memory>\n" + rendered + "\n</memory>\n"

    def _score_importance(self, user: str, assistant: str, llm=None) -> ImportanceScore:
        if llm is None:
            return self._heuristic_importance(user, assistant)
        prompt = (
            "Rate how important this dialogue turn is for future context continuity.\n"
            "Scale 1-10. High score when it contains durable preferences, constraints, goals, "
            "decisions, plans, unresolved tasks, or key facts.\n"
            "Return JSON with fields: score, reason."
        )
        payload = f"User:\n{user}\n\nAssistant:\n{assistant}"
        try:
            scorer = llm.with_structured_output(ImportanceScore)
            result = scorer.invoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ]
            )
            score = max(1, min(int(result.score), 10))
            return ImportanceScore(score=score, reason=result.reason.strip())
        except Exception:
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
            (r"(喜欢|偏好|习惯|兴趣|不喜欢|必须|约束|目标|决定|计划|错误|故障)", 2),
        ]
        for pattern, delta in boosts:
            if re.search(pattern, text):
                score += delta
        score = max(1, min(score, 10))
        return ImportanceScore(score=score, reason="heuristic")

    def _update_global_summary(self, summary: str, turn: dict[str, Any], llm=None) -> str:
        if llm is None:
            return self._fallback_summary(summary, turn)
        prompt = (
            "You maintain a compact running summary for dialogue memory.\n"
            "Update the summary with the new turn while preserving durable facts, decisions, "
            "constraints, open tasks, and user preferences.\n"
            "Keep it concise (<= 1800 chars). Return JSON: {\"summary\": \"...\"}."
        )
        payload = (
            f"Current summary:\n{summary or '(empty)'}\n\n"
            f"New turn:\nUser: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            f"Importance score: {turn.get('importance', 0)}"
        )
        try:
            updater = llm.with_structured_output(SummaryUpdate)
            result = updater.invoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ]
            )
            return (result.summary or "").strip()[:1800]
        except Exception:
            return self._fallback_summary(summary, turn)

    def _silent_turn_decision(
        self,
        *,
        llm,
        summary: str,
        context_snapshot: str,
        latest_turn: dict[str, Any],
    ) -> SilentTurnDecision:
        if llm is None:
            return SilentTurnDecision(
                compressed_summary=self._truncate(context_snapshot, 1600),
                write_daily=int(latest_turn.get("importance", 0)) >= self.importance_threshold,
                daily_note=self._truncate(
                    f"silent: {latest_turn.get('user', '')} | {latest_turn.get('assistant', '')}",
                    260,
                ),
                write_evergreen=False,
                evergreen_note="",
            )
        prompt = (
            "You are running a silent memory-maintenance turn.\n"
            "The context is near capacity. Compress summary and decide if important info should be written "
            "to daily log and/or evergreen diary.\n"
            "Evergreen should only include durable user profile items (preference/habit/interest).\n"
            "Return JSON with fields: compressed_summary, write_daily, daily_note, write_evergreen, evergreen_note."
        )
        payload = (
            f"Current summary:\n{summary or '(empty)'}\n\n"
            f"Latest turn:\nUser: {latest_turn.get('user', '')}\nAssistant: {latest_turn.get('assistant', '')}\n\n"
            f"Context snapshot:\n{self._truncate(context_snapshot, 5000)}"
        )
        try:
            runner = llm.with_structured_output(SilentTurnDecision)
            result = runner.invoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ]
            )
            result.compressed_summary = self._truncate(result.compressed_summary.strip(), 1800)
            result.daily_note = self._truncate(result.daily_note.strip(), 500)
            result.evergreen_note = self._truncate(result.evergreen_note.strip(), 320)
            return result
        except Exception:
            return SilentTurnDecision(
                compressed_summary=self._truncate(context_snapshot, 1600),
                write_daily=False,
                daily_note="",
                write_evergreen=False,
                evergreen_note="",
            )

    def _retrieve_long_term_context(self, *, query_hint: str, max_chars: int) -> str:
        daily_block = self._retrieve_daily_log_block(query_hint=query_hint, max_items=self.daily_retrieval_items)
        evergreen_block = self._retrieve_evergreen_block(
            query_hint=query_hint,
            max_items=self.evergreen_retrieval_items,
        )
        sections = []
        if evergreen_block:
            sections.append("Evergreen diary (non-decaying):\n" + evergreen_block)
        if daily_block:
            sections.append("Daily logs (time-decayed):\n" + daily_block)
        if not sections:
            return ""
        return self._render_with_budget(sections, max_chars=max_chars)

    def _retrieve_daily_log_block(self, *, query_hint: str, max_items: int) -> str:
        files = sorted(self.daily_root.glob("*.json"), reverse=True)
        now = datetime.now(timezone.utc)
        candidates: list[dict[str, Any]] = []
        for file_path in files:
            try:
                day = datetime.strptime(file_path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            age_days = max(0, (now - day).days)
            if age_days > self.daily_lookback_days:
                continue
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            records = self._load_daily_records(raw)
            for record in records:
                summary = str(record.get("summary", "")).strip()
                search_text = self._daily_record_search_text(record)
                if not summary and not search_text:
                    continue
                importance = int(record.get("importance", 5))
                relevance = self._query_relevance(search_text, query_hint)
                decay = self._time_decay(age_days)
                score = (importance / 10.0) * decay + 0.35 * relevance
                candidates.append(
                    {
                        "day": file_path.stem,
                        "record": record,
                        "summary": summary,
                        "score": score,
                        "age_days": age_days,
                        "importance": importance,
                    }
                )
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_items]
        if not ranked:
            return ""
        lines = []
        for item in ranked:
            record = item["record"]
            detail = self._render_daily_record_compact(record)
            lines.append(
                f"- {item['day']} age={item['age_days']}d score={item['score']:.3f}: "
                f"{self._truncate(detail, 320)}"
            )
        return "\n".join(lines)

    def _retrieve_evergreen_block(self, *, query_hint: str, max_items: int) -> str:
        raw = self._load_evergreen()
        entries = []
        for module_name, module in raw.get("modules", {}).items():
            for entry in module.get("entries", []):
                item = dict(entry)
                item["module"] = module_name
                entries.append(item)
        candidates = []
        for entry in entries:
            note = str(entry.get("note", "")).strip()
            if not note:
                continue
            importance = int(entry.get("importance", 7))
            relevance = self._query_relevance(self._evergreen_entry_search_text(entry), query_hint)
            score = (importance / 10.0) + 0.35 * relevance
            candidates.append({"entry": entry, "score": score, "importance": importance})
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_items]
        if not ranked:
            return ""
        return "\n".join(
            f"- {item['entry'].get('module', EVERGREEN_USER_MODULE)} score={item['score']:.3f}: "
            f"{self._truncate(self._render_evergreen_entry(item['entry']), 260)}"
            for item in ranked
        )

    def _append_daily_record(self, *, day: str, record: dict[str, Any]) -> None:
        path = self.daily_root / f"{day}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {
                "schema_version": DAILY_LOG_SCHEMA_VERSION,
                "date": day,
                "records": [],
                "updated_at": _utc_now(),
            }
        records = self._load_daily_records(raw)
        records.append(record)
        raw = {
            "schema_version": DAILY_LOG_SCHEMA_VERSION,
            "date": day,
            "records": records[-600:],
            "updated_at": _utc_now(),
        }
        raw["updated_at"] = _utc_now()
        self._write_json_atomic(path, raw)

    def _load_daily_records(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        return [self._normalize_daily_record(item) for item in raw.get("records", []) if isinstance(item, dict)]

    def _normalize_daily_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("schema_version", DAILY_LOG_SCHEMA_VERSION)
        normalized.setdefault("id", uuid.uuid4().hex)
        normalized.setdefault("ts", _utc_now())
        normalized.setdefault("updated_at", normalized["ts"])
        normalized.setdefault("source", "dialogue_turn")
        normalized.setdefault("thread_id", "default")
        normalized.setdefault("importance", 5)
        normalized.setdefault("category", "conversation")
        normalized.setdefault("confidence", 0.7)
        normalized.setdefault("summary", "")
        normalized.setdefault("facts", [])
        normalized.setdefault("decisions", [])
        normalized.setdefault("tasks", [])
        normalized.setdefault("user_preferences", [])
        normalized.setdefault("constraints", [])
        normalized.setdefault("artifacts", [])
        normalized.setdefault("next_actions", [])
        normalized.setdefault("tags", [])
        normalized.setdefault("raw", {})
        return normalized

    def _build_daily_record(
        self,
        *,
        thread_id: str,
        source: str,
        importance: int,
        decision: DailyLogRecordDecision,
        ts: str,
        turn_id: int | None = None,
        raw_user: str = "",
        raw_assistant: str = "",
    ) -> dict[str, Any]:
        now = _utc_now()
        tags = [self._truncate(item, 40) for item in decision.tags if item.strip()][:8]
        facts = [
            {
                "text": self._truncate(item.strip(), 180),
                "confidence": round(float(decision.confidence), 2),
                "tags": tags,
            }
            for item in decision.facts
            if item.strip()
        ][:8]
        return {
            "schema_version": DAILY_LOG_SCHEMA_VERSION,
            "id": uuid.uuid4().hex,
            "ts": ts,
            "updated_at": now,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "source": source,
            "category": self._normalize_category(decision.category),
            "importance": max(1, min(int(importance), 10)),
            "confidence": round(max(0.0, min(float(decision.confidence), 1.0)), 2),
            "summary": self._truncate(decision.summary.strip(), 500),
            "facts": facts,
            "decisions": [self._truncate(item.strip(), 180) for item in decision.decisions if item.strip()][:6],
            "tasks": [self._truncate(item.strip(), 180) for item in decision.tasks if item.strip()][:6],
            "user_preferences": [
                self._truncate(item.strip(), 180) for item in decision.user_preferences if item.strip()
            ][:6],
            "constraints": [self._truncate(item.strip(), 180) for item in decision.constraints if item.strip()][:6],
            "artifacts": [self._truncate(item.strip(), 180) for item in decision.artifacts if item.strip()][:8],
            "next_actions": [self._truncate(item.strip(), 180) for item in decision.next_actions if item.strip()][:6],
            "tags": tags,
            "raw": {
                "user": self._truncate(raw_user, 700),
                "assistant": self._truncate(raw_assistant, 700),
            },
        }

    def _normalize_category(self, category: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", category.strip().lower()).strip("_")
        return normalized or "conversation"

    def _daily_record_search_text(self, record: dict[str, Any]) -> str:
        parts = [
            record.get("summary", ""),
            " ".join(str(item.get("text", "")) for item in record.get("facts", []) if isinstance(item, dict)),
            " ".join(record.get("decisions", [])),
            " ".join(record.get("tasks", [])),
            " ".join(record.get("user_preferences", [])),
            " ".join(record.get("constraints", [])),
            " ".join(record.get("artifacts", [])),
            " ".join(record.get("next_actions", [])),
            " ".join(record.get("tags", [])),
        ]
        return " ".join(str(item) for item in parts if item).strip()

    def _render_daily_record_compact(self, record: dict[str, Any]) -> str:
        facts = [item.get("text", "") for item in record.get("facts", []) if isinstance(item, dict)]
        parts = [
            f"{record.get('category', 'conversation')}[{record.get('source', 'unknown')}]",
            str(record.get("summary", "")).strip(),
        ]
        if facts:
            parts.append("facts=" + "; ".join(facts[:3]))
        if record.get("tasks"):
            parts.append("tasks=" + "; ".join(record["tasks"][:3]))
        if record.get("decisions"):
            parts.append("decisions=" + "; ".join(record["decisions"][:3]))
        return " | ".join(item for item in parts if item)

    def _append_evergreen_entry(
        self,
        *,
        note: str,
        importance: int,
        source: str,
        thread_id: str,
        module: str = EVERGREEN_USER_MODULE,
        confidence: float = 0.75,
        tags: list[str] | None = None,
        trigger: str = "",
        recommendation: str = "",
    ) -> None:
        raw = self._load_evergreen()
        module = module if module in {EVERGREEN_USER_MODULE, EVERGREEN_REFLECTION_MODULE} else EVERGREEN_USER_MODULE
        entries = raw["modules"][module].setdefault("entries", [])
        normalized = self._normalize_text(note)
        for item in entries:
            if self._normalize_text(str(item.get("note", ""))) == normalized:
                item["importance"] = max(int(item.get("importance", 6)), importance)
                item["confidence"] = max(float(item.get("confidence", 0.0)), confidence)
                item["updated_at"] = _utc_now()
                if recommendation:
                    item["recommendation"] = recommendation
                if trigger:
                    item["trigger"] = trigger
                item["tags"] = sorted(set(item.get("tags", []) + (tags or [])))[:12]
                raw["updated_at"] = _utc_now()
                self._save_evergreen(raw)
                return
        entries.append(
            {
                "id": uuid.uuid4().hex,
                "ts": _utc_now(),
                "updated_at": _utc_now(),
                "thread_id": thread_id,
                "source": source,
                "importance": importance,
                "confidence": round(max(0.0, min(confidence, 1.0)), 2),
                "note": note,
                "trigger": trigger,
                "recommendation": recommendation,
                "tags": tags or [],
            }
        )
        raw["modules"][module]["entries"] = entries[-300:]
        raw["updated_at"] = _utc_now()
        self._save_evergreen(raw)

    def _empty_evergreen(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": EVERGREEN_SCHEMA_VERSION,
            "updated_at": now,
            "modules": {
                EVERGREEN_USER_MODULE: {
                    "description": "Durable user-side facts, preferences, constraints, and goals.",
                    "entries": [],
                },
                EVERGREEN_REFLECTION_MODULE: {
                    "description": "Agent reflexion lessons about strategies, failure modes, and reusable tactics.",
                    "entries": [],
                },
            },
        }

    def _load_evergreen(self) -> dict[str, Any]:
        if not self.evergreen_path.exists():
            return self._empty_evergreen()
        raw = json.loads(self.evergreen_path.read_text(encoding="utf-8"))
        if not isinstance(raw.get("modules"), dict):
            return self._empty_evergreen()
        for module_name in (EVERGREEN_USER_MODULE, EVERGREEN_REFLECTION_MODULE):
            raw["modules"].setdefault(module_name, self._empty_evergreen()["modules"][module_name])
        raw.setdefault("schema_version", EVERGREEN_SCHEMA_VERSION)
        raw.setdefault("updated_at", _utc_now())
        return raw

    def _save_evergreen(self, raw: dict[str, Any]) -> None:
        raw["schema_version"] = EVERGREEN_SCHEMA_VERSION
        raw["updated_at"] = _utc_now()
        self._write_json_atomic(self.evergreen_path, raw)

    def _evergreen_entry_search_text(self, entry: dict[str, Any]) -> str:
        return " ".join(
            str(item)
            for item in [
                entry.get("note", ""),
                entry.get("trigger", ""),
                entry.get("recommendation", ""),
                " ".join(entry.get("tags", [])),
                entry.get("module", ""),
            ]
            if item
        )

    def _render_evergreen_entry(self, entry: dict[str, Any]) -> str:
        parts = [str(entry.get("note", "")).strip()]
        if entry.get("recommendation"):
            parts.append("recommendation=" + str(entry["recommendation"]).strip())
        if entry.get("tags"):
            parts.append("tags=" + ",".join(entry["tags"][:6]))
        return " | ".join(item for item in parts if item)

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _extract_evergreen_note(self, user: str, assistant: str) -> str:
        text = user.strip()
        if not text:
            return ""
        indicator = re.search(
            r"(喜欢|偏好|习惯|兴趣|不喜欢|通常|总是|我会|我一般|I like|I prefer|I usually|I don't like)",
            text,
            re.IGNORECASE,
        )
        if not indicator:
            return ""
        return self._truncate(text, 260)

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

    def _build_reflection_decision(
        self,
        *,
        user: str,
        assistant: str,
        tool_observations: list[str],
        llm,
    ) -> ReflectionDecision | None:
        prompt = (
            "You are Falco's reflexion module. Extract one reusable operational lesson from the latest turn.\n"
            "Write only if the lesson will improve future agent behavior, tool choice, validation, planning, "
            "or error recovery. Do not store user private facts here; those belong to user memory.\n"
            "Return JSON: should_write, lesson, trigger, recommendation, confidence, tags.\n"
            "Keep lesson and recommendation concise."
        )
        observations = "\n".join(f"- {self._truncate(item, 500)}" for item in tool_observations[:8])
        payload = (
            f"User:\n{self._truncate(user, 1200)}\n\n"
            f"Assistant:\n{self._truncate(assistant, 1200)}\n\n"
            f"Tool observations:\n{observations or '(none)'}"
        )
        try:
            reflector = llm.with_structured_output(ReflectionDecision)
            result = reflector.invoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ]
            )
            result.lesson = self._truncate(result.lesson.strip(), 320)
            result.trigger = self._truncate(result.trigger.strip(), 240)
            result.recommendation = self._truncate(result.recommendation.strip(), 260)
            result.tags = [self._truncate(item.strip(), 40) for item in result.tags if item.strip()][:8]
            return result
        except Exception:
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
            prompt = (
                "Decide whether this dialogue turn should be written into a structured daily memory log.\n"
                "Extract durable, useful records instead of copying the chat.\n"
                "Write only if it contains future-useful facts, decisions, tasks, preferences, constraints, "
                "artifacts, or next actions.\n"
                "Return JSON with fields: should_write, summary, category, confidence, facts, decisions, "
                "tasks, user_preferences, constraints, artifacts, next_actions, tags.\n"
                "Keep every list item concise and atomic."
            )
            payload = (
                f"Importance: {importance}\n"
                f"User:\n{user}\n\n"
                f"Assistant:\n{assistant}"
            )
            try:
                extractor = llm.with_structured_output(DailyLogRecordDecision)
                decision = extractor.invoke(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": payload},
                    ]
                )
                decision.summary = self._truncate(decision.summary.strip(), 500)
                decision.facts = [self._truncate(item.strip(), 180) for item in decision.facts if item.strip()][:8]
                decision.decisions = [
                    self._truncate(item.strip(), 180) for item in decision.decisions if item.strip()
                ][:6]
                decision.tasks = [self._truncate(item.strip(), 180) for item in decision.tasks if item.strip()][:6]
                decision.user_preferences = [
                    self._truncate(item.strip(), 180) for item in decision.user_preferences if item.strip()
                ][:6]
                decision.constraints = [
                    self._truncate(item.strip(), 180) for item in decision.constraints if item.strip()
                ][:6]
                decision.artifacts = [
                    self._truncate(item.strip(), 180) for item in decision.artifacts if item.strip()
                ][:8]
                decision.next_actions = [
                    self._truncate(item.strip(), 180) for item in decision.next_actions if item.strip()
                ][:6]
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
                r"(是|在|包含|版本|时间|日期|地址|电话|邮箱|deadline|due|version|date|must|constraint|prefer)",
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
            r"\b(prefer|like|usually|style|habit)\b|希望|偏好|喜欢",
        )
        decisions = self._heuristic_extract_by_signal(
            joined,
            r"\b(decide|decision|agreed|chosen|final)\b|确定|决定",
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
        lines = re.split(r"[。！？\n]|(?<=[.!?])\s+", text)
        facts: list[str] = []
        for line in lines:
            piece = line.strip()
            if len(piece) < 6:
                continue
            if re.search(
                r"(是|在|包含|需要|必须|偏好|习惯|兴趣|版本|日期|时间|地址|deadline|version|date|must|prefer|usually)",
                piece,
                re.IGNORECASE,
            ):
                facts.append(self._truncate(piece, 160))
            if len(facts) >= 6:
                break
        return facts

    def _heuristic_extract_by_signal(self, text: str, pattern: str, limit: int = 4) -> list[str]:
        lines = re.split(r"[\n。；;]|(?<=[.!?])\s+", text)
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
            return "user_preference"
        if constraints:
            return "constraint"
        if tasks or re.search(r"\b(todo|task|deadline|due|implement|fix|deploy|test)\b", low):
            return "task"
        if re.search(r"\b(decision|decide|agreed|final)\b", low):
            return "decision"
        if re.search(r"\b(error|bug|failed|incident|regression)\b", low):
            return "issue"
        return "conversation"

    def _infer_daily_tags(self, text: str) -> list[str]:
        tags: list[str] = []
        signals = [
            ("task", r"\b(todo|task|deadline|due|implement|fix)\b"),
            ("preference", r"\b(prefer|like|usually|style|habit)\b|偏好|喜欢"),
            ("constraint", r"\b(must|never|always|constraint|requirement)\b"),
            ("bug", r"\b(error|bug|failed|incident|regression)\b"),
            ("architecture", r"\b(api|schema|architecture|database|agent|memory|rag)\b"),
        ]
        for tag, pattern in signals:
            if re.search(pattern, text, re.IGNORECASE):
                tags.append(tag)
        return tags[:6]

    def _query_relevance(self, text: str, query_hint: str) -> float:
        query_tokens = self._tokenize(query_hint)
        if not query_tokens:
            return 0.0
        text_low = text.lower()
        hit = 0
        for token in query_tokens:
            if token in text_low:
                hit += 1
        return hit / max(1, len(query_tokens))

    def _tokenize(self, text: str) -> list[str]:
        if not text.strip():
            return []
        tokens = re.findall(r"[\u4e00-\u9fff]{1,8}|[a-zA-Z0-9_]{2,}", text.lower())
        dedup = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            dedup.append(token)
        return dedup[:20]

    def _time_decay(self, age_days: int) -> float:
        half_life = max(1, self.daily_half_life_days)
        return math.exp(-math.log(2) * (age_days / half_life))

    def _fallback_summary(self, summary: str, turn: dict[str, Any]) -> str:
        user = self._truncate(turn.get("user", ""), 220)
        assistant = self._truncate(turn.get("assistant", ""), 220)
        merged = summary.strip()
        appendix = f"- U: {user}\n- A: {assistant}"
        if merged:
            merged = f"{merged}\n{appendix}"
        else:
            merged = appendix
        return merged[-1800:]

    def _history_to_turns(self, history: list[dict]) -> list[dict]:
        turns: list[dict] = []
        pending_user = ""
        pending_ts = None
        turn_id = 0
        for item in history:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            ts = item.get("ts", _utc_now())
            if role == "user":
                pending_user = content
                pending_ts = ts
                continue
            if role == "assistant":
                turn_id += 1
                turns.append(
                    {
                        "id": turn_id,
                        "ts": pending_ts or ts,
                        "user": pending_user,
                        "assistant": content,
                        "importance": 5,
                        "importance_reason": "migrated",
                        "is_key": False,
                    }
                )
                pending_user = ""
                pending_ts = None
        return turns[-self.max_rounds :]

    def _render_with_budget(self, sections: list[str], *, max_chars: int) -> str:
        remain = max(0, max_chars)
        selected: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) + 2 <= remain:
                selected.append(section)
                remain -= len(section) + 2
                continue
            if remain <= 120:
                break
            selected.append(self._truncate(section, remain))
            break
        return "\n\n".join(selected).strip()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _truncate(self, text: str, limit: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= limit:
            return clean
        if limit <= 3:
            return clean[:limit]
        return clean[: limit - 3] + "..."
