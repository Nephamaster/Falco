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

## Config

Use `config.yaml`:

```yaml
memory:
  root: ./.falco/memory
  context_soft_limit_tokens: 7000
  context_max_tokens: 9000
  silent_turn_cooldown_rounds: 4
  daily_half_life_days: 30
  daily_lookback_days: 180
  daily_retrieval_items: 8
  evergreen_retrieval_items: 5
```

`memory.root` can be absolute or relative path.
