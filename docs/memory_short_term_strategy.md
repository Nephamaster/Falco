# Short-term Memory Strategy (Implemented)

## Strategy

- Recent window: keep latest `N` dialogue rounds.
- Global summary: maintain an incremental running summary.
- Importance filtering: each new round is scored by LLM (`1-10`) and high-value rounds are retained as key history.

## How context is assembled

Falco memory context now includes:

1. Global conversation summary
2. Known memory facts
3. Key historical turns (importance-ranked, excluding recent window)
4. Recent conversation turns (`N` rounds)

## Config

```bash
FALCO_MEMORY_RECENT_ROUNDS=6
FALCO_MEMORY_KEY_ROUNDS=4
FALCO_MEMORY_IMPORTANCE_THRESHOLD=7
FALCO_MEMORY_MAX_ROUNDS=160
```

## Storage

Thread memory files remain at:

`/.falco/memory/<thread_id>.json`

New fields:

- `turns`: round-based memory entries with importance metadata
- `global_summary`: compact rolling summary
