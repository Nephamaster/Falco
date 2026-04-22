from __future__ import annotations

import json
import math
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken

from harness.agents.memory.runtime import utc_now


class MemoryUtilityMixin:
    def _tokenizer_model(self) -> str:
        return getattr(self, "tokenizer_model", "gpt-4o-mini")

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_encoding_for_model(model_name: str):
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _get_encoding(self, model_name: str):
        return self._get_encoding_for_model(model_name)

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        encoding = self._get_encoding(self._tokenizer_model())
        return len(encoding.encode(text))

    def _truncate_tail(self, text: str, limit_tokens: int) -> str:
        clean = " ".join((text or "").split())
        if not clean:
            return ""
        if limit_tokens <= 0:
            return ""
        encoding = self._get_encoding(self._tokenizer_model())
        tokens = encoding.encode(clean)
        if len(tokens) <= limit_tokens:
            return clean
        return encoding.decode(tokens[-limit_tokens:])

    def _path(self, thread_id: str) -> Path:
        safe = "".join(ch for ch in thread_id if ch.isalnum() or ch in ("-", "_")) or "default"
        return self.root / f"{safe}.json"

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

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

    def _history_to_turns(self, history: list[dict]) -> list[dict]:
        turns: list[dict] = []
        pending_user_parts: list[str] = []
        pending_assistant_parts: list[str] = []
        pending_ts = None
        turn_id = 0

        def flush_pending() -> None:
            nonlocal turn_id, pending_user_parts, pending_assistant_parts, pending_ts
            if not pending_user_parts and not pending_assistant_parts:
                pending_ts = None
                return
            turn_id += 1
            turns.append(
                {
                    "id": turn_id,
                    "ts": pending_ts or utc_now(),
                    "user": "\n".join(part for part in pending_user_parts if part).strip(),
                    "assistant": "\n".join(part for part in pending_assistant_parts if part).strip(),
                    "importance": 5,
                    "importance_reason": "migrated",
                    "is_key": False,
                }
            )
            pending_user_parts = []
            pending_assistant_parts = []
            pending_ts = None

        for item in history:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            ts = item.get("ts", utc_now())
            if role == "user":
                if pending_assistant_parts:
                    flush_pending()
                if pending_ts is None:
                    pending_ts = ts
                pending_user_parts.append(content)
                continue
            if role == "assistant":
                if pending_ts is None:
                    pending_ts = ts
                pending_assistant_parts.append(content)
                continue
            flush_pending()

        flush_pending()
        return turns[-self.max_rounds :]

    def _render_with_budget(self, sections: list[str], *, max_tokens: int) -> str:
        remain = max(0, max_tokens)
        selected: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            section_tokens = self._count_tokens(section)
            sep_tokens = 1
            if section_tokens + sep_tokens <= remain:
                selected.append(section)
                remain -= section_tokens + sep_tokens
                continue
            if remain <= 20:
                break
            selected.append(self._truncate(section, remain))
            break
        return "\n\n".join(selected).strip()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _truncate(self, text: str, limit: int) -> str:
        clean = " ".join((text or "").split())
        if not clean:
            return ""
        if limit <= 0:
            return ""
        encoding = self._get_encoding(self._tokenizer_model())
        tokens = encoding.encode(clean)
        if len(tokens) <= limit:
            return clean
        if limit <= 3:
            return encoding.decode(tokens[:limit])
        return encoding.decode(tokens[: limit - 1]).rstrip() + "..."
