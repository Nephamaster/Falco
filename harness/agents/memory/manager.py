from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
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


class DailyLogDecision(BaseModel):
    should_write: bool = Field(default=False)
    note: str = Field(default="")
    facts: list[str] = Field(default_factory=list)


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

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.daily_root.mkdir(parents=True, exist_ok=True)
        if not self.evergreen_path.exists():
            self.evergreen_path.write_text(
                json.dumps({"entries": [], "updated_at": _utc_now()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
        self._path(thread_id).write_text(
            json.dumps(memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_fact(self, thread_id: str, note: str) -> None:
        memory = self.load(thread_id)
        facts = memory.setdefault("facts", [])
        facts.append({"ts": _utc_now(), "note": note.strip()})
        memory["facts"] = facts[-self.max_facts :]
        self.save(thread_id, memory)

    def append_turn(self, thread_id: str, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
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

        daily_note = self._extract_daily_log_note(
            user=user_text,
            assistant=assistant_text,
            importance=int(turn["importance"]),
            llm=llm,
        )
        if daily_note is not None:
            self._append_daily_entry(
                day=_today_str(),
                entry={
                    "ts": turn["ts"],
                    "thread_id": thread_id,
                    "type": "round_valuable",
                    "importance": int(turn["importance"]),
                    "note": self._truncate(daily_note["note"], 500),
                    "facts": [self._truncate(item, 160) for item in daily_note.get("facts", [])][:6],
                },
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
            self._append_daily_entry(
                day=_today_str(),
                entry={
                    "ts": _utc_now(),
                    "thread_id": thread_id,
                    "type": "silent",
                    "importance": max(self.importance_threshold, int(turns[-1].get("importance", 7))),
                    "note": self._truncate(decision.daily_note.strip(), 500),
                },
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
            entries = raw.get("entries", [])
            for entry in entries:
                note = str(entry.get("note", "")).strip()
                if not note:
                    continue
                importance = int(entry.get("importance", 5))
                relevance = self._query_relevance(note, query_hint)
                decay = self._time_decay(age_days)
                score = (importance / 10.0) * decay + 0.35 * relevance
                candidates.append(
                    {
                        "day": file_path.stem,
                        "note": note,
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
            lines.append(
                f"- {item['day']} age={item['age_days']}d score={item['score']:.3f}: "
                f"{self._truncate(item['note'], 260)}"
            )
        return "\n".join(lines)

    def _retrieve_evergreen_block(self, *, query_hint: str, max_items: int) -> str:
        raw = json.loads(self.evergreen_path.read_text(encoding="utf-8"))
        entries = raw.get("entries", [])
        candidates = []
        for entry in entries:
            note = str(entry.get("note", "")).strip()
            if not note:
                continue
            importance = int(entry.get("importance", 7))
            relevance = self._query_relevance(note, query_hint)
            score = (importance / 10.0) + 0.35 * relevance
            candidates.append({"note": note, "score": score, "importance": importance})
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_items]
        if not ranked:
            return ""
        return "\n".join(
            f"- score={item['score']:.3f}: {self._truncate(item['note'], 220)}"
            for item in ranked
        )

    def _append_daily_entry(self, *, day: str, entry: dict[str, Any]) -> None:
        path = self.daily_root / f"{day}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {"date": day, "entries": [], "updated_at": _utc_now()}
        entries = raw.get("entries", [])
        entries.append(entry)
        raw["entries"] = entries[-600:]
        raw["updated_at"] = _utc_now()
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_evergreen_entry(self, *, note: str, importance: int, source: str, thread_id: str) -> None:
        raw = json.loads(self.evergreen_path.read_text(encoding="utf-8"))
        entries = raw.get("entries", [])
        normalized = self._normalize_text(note)
        for item in entries:
            if self._normalize_text(str(item.get("note", ""))) == normalized:
                item["importance"] = max(int(item.get("importance", 6)), importance)
                item["updated_at"] = _utc_now()
                raw["updated_at"] = _utc_now()
                self.evergreen_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                return
        entries.append(
            {
                "ts": _utc_now(),
                "updated_at": _utc_now(),
                "thread_id": thread_id,
                "source": source,
                "importance": importance,
                "note": note,
            }
        )
        raw["entries"] = entries[-300:]
        raw["updated_at"] = _utc_now()
        self.evergreen_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _extract_daily_log_note(
        self,
        *,
        user: str,
        assistant: str,
        importance: int,
        llm=None,
    ) -> dict[str, Any] | None:
        if llm is not None:
            prompt = (
                "Decide whether this turn should be written into a daily memory log.\n"
                "Write only if it has valuable/important information or extractable facts.\n"
                "Return JSON with: should_write, note, facts.\n"
                "facts should be concise atomic facts."
            )
            payload = (
                f"Importance: {importance}\n"
                f"User: {user}\n"
                f"Assistant: {assistant}"
            )
            try:
                extractor = llm.with_structured_output(DailyLogDecision)
                decision = extractor.invoke(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": payload},
                    ]
                )
                facts = [self._truncate(item.strip(), 160) for item in decision.facts if item.strip()][:6]
                if decision.should_write and (decision.note.strip() or facts):
                    return {
                        "note": decision.note.strip() or "Valuable turn captured.",
                        "facts": facts,
                    }
            except Exception:
                pass
        return self._extract_daily_log_note_heuristic(user=user, assistant=assistant, importance=importance)

    def _extract_daily_log_note_heuristic(
        self,
        *,
        user: str,
        assistant: str,
        importance: int,
    ) -> dict[str, Any] | None:
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
        note = self._truncate(joined, 340)
        return {"note": note, "facts": facts}

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
