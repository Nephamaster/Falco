from __future__ import annotations

from functools import lru_cache

import tiktoken


DEFAULT_TOKENIZER_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=8)
def get_encoding(model_name: str = DEFAULT_TOKENIZER_MODEL):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, *, model_name: str = DEFAULT_TOKENIZER_MODEL) -> int:
    if not text:
        return 0
    return len(get_encoding(model_name).encode(text))


def truncate_tokens(text: str, limit_tokens: int, *, model_name: str = DEFAULT_TOKENIZER_MODEL) -> str:
    if not text:
        return ""
    if limit_tokens <= 0:
        return ""
    encoding = get_encoding(model_name)
    tokens = encoding.encode(text)
    if len(tokens) <= limit_tokens:
        return text
    return encoding.decode(tokens[:limit_tokens])
