# Memory Optimization Phase 2 (OpenClaw-aligned subset)

Implemented items from OpenClaw "four-stage retrieval flow" scope requested in this task:

- Long-term memory split:
  - Daily logs: `.falco/memory/daily/YYYY-MM-DD.json`
  - Evergreen diary: `.falco/memory/evergreen.json`
- Time decay on daily logs during context injection:
  - `decay = exp(-ln(2) * age_days / half_life_days)`
  - Default `half_life_days = 30`
- Context compression + silent turn:
  - When memory context size is near limit, system runs a hidden maintenance turn.
  - It compresses global summary and may write critical notes to daily/evergreen memory.
  - Writes are optional (not forced every turn).

## Retrieval behavior

- Evergreen notes are non-decaying and ranked by `importance + query relevance`.
- Daily notes are ranked by `importance * time_decay + query relevance`.
- Retrieved long-term memory is injected into the memory context with budget control.

## New environment variables

```bash
FALCO_MEMORY_ROOT=./.falco/memory
FALCO_MEMORY_CONTEXT_SOFT_LIMIT_CHARS=7000
FALCO_MEMORY_CONTEXT_MAX_CHARS=9000
FALCO_MEMORY_SILENT_TURN_COOLDOWN_ROUNDS=4
FALCO_MEMORY_DAILY_HALF_LIFE_DAYS=30
FALCO_MEMORY_DAILY_LOOKBACK_DAYS=180
FALCO_MEMORY_DAILY_RETRIEVAL_ITEMS=8
FALCO_MEMORY_EVERGREEN_RETRIEVAL_ITEMS=5
```

`FALCO_MEMORY_ROOT` can be absolute or relative path.
