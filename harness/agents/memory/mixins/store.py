from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.agents.memory.models import (
    DAILY_LOG_ALLOWED_CATEGORIES,
    DAILY_LOG_SCHEMA_VERSION,
    EVERGREEN_REFLECTION_MODULE,
    EVERGREEN_SCHEMA_VERSION,
    EVERGREEN_USER_MODULE,
    DailyLogRecordDecision,
)
from harness.agents.memory.runtime import parse_ts, utc_now


class MemoryStoreMixin:
    _CATEGORY_ALIASES = {
        "conversation": "other",
        "general": "other",
        "misc": "other",
        "memory_maintenance": "other",
        "maintenance": "other",
        "user_preference": "preference",
        "preferences": "preference",
        "fact": "info",
        "facts": "info",
        "issue": "info",
        "problem": "info",
        "error": "info",
        "artifact_update": "artifact",
    }

    @property
    def daily_root(self) -> Path:
        return self.root / "daily"

    @property
    def evergreen_path(self) -> Path:
        return self.root / "evergreen.json"

    def load(self, thread_id: str) -> dict:
        path = self._path(thread_id)
        if not path.exists():
            return {
                "thread_id": thread_id,
                "updated_at": utc_now(),
                "facts": [],
                "history": [],
                "turns": [],
                "global_summary": "",
                "last_silent_turn_id": 0,
                "next_turn_id": 1,
                "pending_evicted_turns": [],
            }
        memory = json.loads(path.read_text(encoding="utf-8"))
        memory.setdefault("facts", [])
        memory.setdefault("history", [])
        memory.setdefault("turns", [])
        memory.setdefault("global_summary", "")
        memory.setdefault("last_silent_turn_id", 0)
        memory.setdefault("pending_evicted_turns", [])
        if not memory["turns"] and memory["history"]:
            memory["turns"] = self._history_to_turns(memory["history"])
        if not isinstance(memory.get("pending_evicted_turns"), list):
            memory["pending_evicted_turns"] = []
        max_turn_id = max((int(item.get("id", 0)) for item in memory["turns"]), default=0)
        next_turn_id = int(memory.get("next_turn_id", 0) or 0)
        if next_turn_id <= max_turn_id:
            next_turn_id = max_turn_id + 1
        memory["next_turn_id"] = max(1, next_turn_id)
        return memory

    def save(self, thread_id: str, memory: dict) -> None:
        memory["updated_at"] = utc_now()
        self._write_json_atomic(self._path(thread_id), memory)

    def add_fact(self, thread_id: str, note: str) -> None:
        with self._lock:
            memory = self.load(thread_id)
            facts = memory.setdefault("facts", [])
            facts.append({"ts": utc_now(), "note": note.strip()})
            memory["facts"] = facts[-self.max_facts :]
            self.save(thread_id, memory)

    def append_turn(self, thread_id: str, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        with self._lock:
            memory = self.load(thread_id)
            history = memory.setdefault("history", [])
            history.append({"ts": utc_now(), "role": role, "content": text})
            memory["history"] = history[-self.max_history :]
            self.save(thread_id, memory)

    def _retrieve_long_term_context(self, *, query_hint: str, max_tokens: int) -> str:
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
        return self._render_with_budget(sections, max_tokens=max_tokens)

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
        if not candidates:
            return ""
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_items]
        lines = []
        for item in ranked:
            lines.append(
                f"- {item['day']} score={item['score']:.3f} imp={item['importance']} "
                f"age={item['age_days']}d: {self._render_daily_record_compact(item['record'])}"
            )
        return "\n".join(lines)

    def _retrieve_evergreen_block(self, *, query_hint: str, max_items: int) -> str:
        raw = self._load_evergreen()
        modules = raw.get("modules", {})
        candidates: list[dict[str, Any]] = []
        for module_name, module_data in modules.items():
            entries = module_data.get("entries", [])
            for entry in entries:
                entry = dict(entry)
                entry["module"] = module_name
                text = self._evergreen_entry_search_text(entry)
                if not text:
                    continue
                importance = int(entry.get("importance", 5))
                relevance = self._query_relevance(text, query_hint)
                score = (importance / 10.0) + 0.4 * relevance
                candidates.append({"entry": entry, "score": score})
        if not candidates:
            return ""
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_items]
        return "\n".join(
            [
                f"- {item['entry'].get('module', EVERGREEN_USER_MODULE)} score={item['score']:.3f}: "
                f"{self._render_evergreen_entry(item['entry'])}"
                for item in ranked
            ]
        )

    def _append_daily_record(self, *, day: str, record: dict[str, Any]) -> None:
        path = self.daily_root / f"{day}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {
                "schema_version": DAILY_LOG_SCHEMA_VERSION,
                "day": day,
                "updated_at": utc_now(),
                "records": [],
            }
        records = self._load_daily_records(raw)
        records.append(self._normalize_daily_record(record))
        raw["records"] = records[-500:]
        raw["updated_at"] = utc_now()
        raw["schema_version"] = DAILY_LOG_SCHEMA_VERSION
        raw["day"] = day
        self._write_json_atomic(path, raw)

    def _load_daily_records(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        records = raw.get("records", [])
        if isinstance(records, list):
            return [self._normalize_daily_record(item) for item in records if isinstance(item, dict)]
        return []

    def _normalize_daily_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("schema_version", DAILY_LOG_SCHEMA_VERSION)
        normalized.setdefault("id", uuid.uuid4().hex)
        normalized.setdefault("ts", utc_now())
        normalized.setdefault("updated_at", normalized["ts"])
        normalized.setdefault("source", "dialogue_turn")
        normalized.setdefault("thread_id", "default")
        normalized.setdefault("importance", 5)
        normalized.setdefault("category", "other")
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
        for key, limit in [
            ("facts", 8),
            ("decisions", 6),
            ("tasks", 6),
            ("user_preferences", 6),
            ("constraints", 6),
            ("artifacts", 8),
            ("next_actions", 6),
            ("tags", 8),
        ]:
            normalized[key] = self._normalize_daily_list_field(normalized.get(key), limit=limit)
        normalized["category"] = self._normalize_category(str(normalized.get("category", "other")))
        normalized["importance"] = max(1, min(int(normalized.get("importance", 5)), 10))
        normalized["confidence"] = round(max(0.0, min(float(normalized.get("confidence", 0.7)), 1.0)), 2)
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
        return {
            "schema_version": DAILY_LOG_SCHEMA_VERSION,
            "id": uuid.uuid4().hex,
            "ts": ts,
            "updated_at": ts,
            "source": source,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "importance": max(1, min(int(importance), 10)),
            "category": self._normalize_category(decision.category),
            "confidence": round(max(0.0, min(float(decision.confidence), 1.0)), 2),
            "summary": self._truncate(decision.summary.strip(), 500),
            "facts": decision.facts[:8],
            "decisions": decision.decisions[:6],
            "tasks": decision.tasks[:6],
            "user_preferences": decision.user_preferences[:6],
            "constraints": decision.constraints[:6],
            "artifacts": decision.artifacts[:8],
            "next_actions": decision.next_actions[:6],
            "tags": decision.tags[:8],
            "raw": {
                "user": self._truncate(raw_user, 500),
                "assistant": self._truncate(raw_assistant, 500),
            }
            if raw_user or raw_assistant
            else {},
        }

    def _normalize_category(self, category: str) -> str:
        normalized = self._normalize_text(category).replace(" ", "_")
        if not normalized:
            return "other"
        normalized = self._CATEGORY_ALIASES.get(normalized, normalized)
        if normalized not in DAILY_LOG_ALLOWED_CATEGORIES:
            return "other"
        return normalized

    def _daily_record_search_text(self, record: dict[str, Any]) -> str:
        facts = self._normalize_daily_list_field(record.get("facts"), limit=12)
        decisions = self._normalize_daily_list_field(record.get("decisions"), limit=10)
        tasks = self._normalize_daily_list_field(record.get("tasks"), limit=10)
        user_preferences = self._normalize_daily_list_field(record.get("user_preferences"), limit=10)
        constraints = self._normalize_daily_list_field(record.get("constraints"), limit=10)
        artifacts = self._normalize_daily_list_field(record.get("artifacts"), limit=12)
        next_actions = self._normalize_daily_list_field(record.get("next_actions"), limit=10)
        tags = self._normalize_daily_list_field(record.get("tags"), limit=12)
        parts = [
            record.get("summary", ""),
            " ".join(facts),
            " ".join(decisions),
            " ".join(tasks),
            " ".join(user_preferences),
            " ".join(constraints),
            " ".join(artifacts),
            " ".join(next_actions),
            " ".join(tags),
            f"{record.get('category', 'other')}[{record.get('source', 'unknown')}]",
        ]
        return " ".join(str(part) for part in parts if part).strip()

    def _render_daily_record_compact(self, record: dict[str, Any]) -> str:
        base = str(record.get("summary", "")).strip()
        if not base:
            pieces = []
            for key in ("decisions", "tasks", "facts", "constraints", "next_actions"):
                values = [
                    self._truncate(item, 120)
                    for item in self._normalize_daily_list_field(record.get(key), limit=2)
                    if str(item).strip()
                ]
                if values:
                    pieces.append(f"{key}={'; '.join(values)}")
            base = " | ".join(pieces)
        tags = ",".join(self._normalize_daily_list_field(record.get("tags"), limit=5))
        if tags:
            base = f"{base} [tags:{tags}]".strip()
        return self._truncate(base or "(empty)", 260)

    def _normalize_daily_list_field(self, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = self._coerce_record_item_to_text(item)
            if not text:
                continue
            items.append(self._truncate(text, 220))
            if len(items) >= limit:
                break
        return items

    def _coerce_record_item_to_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, (int, float, bool)):
            return str(item)
        if isinstance(item, dict):
            # Backward compatibility: older records may store structured objects in list fields.
            for key in ("summary", "note", "value", "title", "name", "content", "text"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            values = [str(v).strip() for v in item.values() if str(v).strip()]
            return " | ".join(values[:3])
        return str(item).strip()

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
        note = self._truncate(note.strip(), 320)
        if not note:
            return
        module = module if module in {EVERGREEN_USER_MODULE, EVERGREEN_REFLECTION_MODULE} else EVERGREEN_USER_MODULE
        raw = self._load_evergreen()
        entries = raw["modules"][module].setdefault("entries", [])
        key = self._normalize_text(note)
        for idx, entry in enumerate(entries):
            if self._normalize_text(str(entry.get("note", ""))) == key:
                merged = dict(entry)
                merged["updated_at"] = utc_now()
                merged["importance"] = max(int(merged.get("importance", 5)), importance)
                merged["confidence"] = round(max(0.0, min(max(float(merged.get("confidence", 0.7)), confidence), 1.0)), 2)
                merged["source"] = source
                merged["thread_id"] = thread_id
                if trigger:
                    merged["trigger"] = self._truncate(trigger, 240)
                if recommendation:
                    merged["recommendation"] = self._truncate(recommendation, 260)
                if tags:
                    merged["tags"] = list({*(merged.get("tags", [])), *tags})[:8]
                entries[idx] = merged
                raw["updated_at"] = utc_now()
                self._save_evergreen(raw)
                return
        entries.append(
            {
                "id": uuid.uuid4().hex,
                "ts": utc_now(),
                "updated_at": utc_now(),
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
        raw["updated_at"] = utc_now()
        self._save_evergreen(raw)

    def _empty_evergreen(self) -> dict[str, Any]:
        now = utc_now()
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
        raw.setdefault("updated_at", utc_now())
        return raw

    def _save_evergreen(self, raw: dict[str, Any]) -> None:
        raw["schema_version"] = EVERGREEN_SCHEMA_VERSION
        raw["updated_at"] = utc_now()
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
