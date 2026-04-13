from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pathlib import Path
from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImportanceScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = Field(default="")


class SummaryUpdate(BaseModel):
    summary: str = Field(description="Updated compact global summary.")


@dataclass
class ConversationMemoryManager:
    root: Path
    max_history: int = 60
    max_facts: int = 50
    recent_rounds: int = 6
    key_rounds: int = 4
    importance_threshold: int = 7
    max_rounds: int = 160

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

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
            }
        memory = json.loads(path.read_text(encoding="utf-8"))
        memory.setdefault("facts", [])
        memory.setdefault("history", [])
        memory.setdefault("turns", [])
        memory.setdefault("global_summary", "")
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
        self.save(thread_id, memory)

    def build_context_block(
        self,
        thread_id: str,
        max_items: int = 12,
        *,
        recent_rounds: int | None = None,
        key_rounds: int | None = None,
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
        selected_key = key_candidates[:key_n] if key_n > 0 else []
        selected_key = sorted(selected_key, key=lambda item: int(item.get("id", 0)))

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

        return "\n\n".join(section.strip() for section in sections if section.strip()).strip()

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

    def _truncate(self, text: str, limit: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3] + "..."
