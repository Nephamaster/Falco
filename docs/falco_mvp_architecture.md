# Falco MVP Architecture (DeerFlow-Inspired)

## 1) Design Mapping from DeerFlow

- Lead agent as the single planner/executor entrypoint.
- Tool-first execution loop via LangGraph (`lead_agent -> tools -> lead_agent`).
- Skill cards injected into the lead system prompt (runtime composability).
- Tool-as-Agent paradigm via `delegate_task`:
  - Sub-agent has isolated message context.
  - Sub-agent can access a bounded tool subset.
- Context isolation and scarcity management:
  - Thread memory is persisted separately from live message state.
  - Prompt only receives compact memory block and enabled skills.

## 2) MVP Modules

- `falco/orchestrator.py`
  - Builds the LangGraph workflow.
  - Hydrates memory + skills context.
  - Executes lead agent and tool loops.
  - Persists compact conversation memory.
- `falco/tools.py`
  - Filesystem tools (`list/read/write/search`).
  - Memory tools (`add_memory/query_memory`).
  - Skill tools (`skill_catalog/skill_manage`).
  - Delegation tool (`delegate_task`) for isolated sub-agent execution.
- `falco/subagent.py`
  - Lightweight isolated worker agent loop with tool calls.
- `falco/skills.py`
  - Skill storage, parsing, enable/disable, prompt assembly.
- `falco/memory.py`
  - Thread-scoped durable memory files and compact context rendering.

## 3) Runtime Flow

1. User message enters graph with `thread_id`.
2. `hydrate_context` loads:
   - memory block from `.falco/memory/<thread>.json`
   - active skills from `.falco/skills/*.md`
3. `lead_agent` reasons and decides:
   - answer directly, or
   - call tools, including delegated sub-agent.
4. Tool outputs return to lead agent until no further tool call.
5. `persist` stores last user + assistant turns for next invocation.

## 4) Why This Is the Minimum Viable Set

- Autonomous orchestration: lead agent dynamically chooses tools and delegation.
- Memory/context management: thread-level persistence + compact prompt hydration.
- Toolset construction: modular registry with filesystem/memory/skills/delegation.
- Skill management and usage: dynamic enable/disable/create/update plus prompt-time injection.
