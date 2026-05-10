# Falco Evaluation Harness

This directory contains an offline eval runner for the Falco agent system.

The runner is intentionally separate from the runtime. It calls the existing
`FalcoOrchestrator` entrypoints and reconstructs tool traces from LangGraph
messages.

## Validate cases without calling the model

```bash
python -m evaluation.runners.run_eval --suite smoke --dry-run
```

## Run a suite

```bash
python -m evaluation.runners.run_eval --suite smoke --no-llm-judge
```

Results are written to:

```text
evaluation/runs/<timestamp>/results.jsonl
evaluation/runs/<timestamp>/summary.md
```

## Case format

Each suite is a JSONL file under `evaluation/cases`.

```json
{
  "id": "example_001",
  "category": "tool_use",
  "input": "Read README.md and summarize it.",
  "expected_tools": ["read_file"],
  "forbidden_tools": ["write_file"],
  "max_tool_calls": 3,
  "must_include": ["Falco"],
  "rubric": {
    "judge_dimensions": ["correctness", "tool_quality"]
  }
}
```

For multi-turn cases, add `resume_steps`. Steps call `invoke()` by default.
Set `"resume": true` only when the previous turn produced a HITL interrupt.

Supported automatic checks:

- `must_include` and `must_not_include`
- `expected_tools`, `forbidden_tools`, and `max_tool_calls`
- minimal JSON shape checks through `expected_json_schema`
- RAG-use check for `category="rag"` or `rubric.requires_rag=true`
- optional LLM judge when `LLM_API_KEY` is set and `--no-llm-judge` is not used
