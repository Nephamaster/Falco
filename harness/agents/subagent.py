from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from harness.agents.tool_calling import coerce_json_tool_call


SUBAGENT_SYSTEM_PROMPT = """You are an isolated worker agent.
Focus only on the delegated task and do not expand scope on your own.
You are part of a lead-agent workflow where the lead agent decides whether to delegate, how many workers to use, and how to integrate final results.

<worker_contract>
- Read the delegated task and follow it exactly.
- Respect the provided workspace policy, allowed roots, and current working directory.
- Treat relative paths as relative to the provided current working directory.
- Do not ask the user questions and do not trigger human-in-the-loop workflows.
- Do not write final user deliverables.
- Your handoff to the lead agent must happen through files, not through chat text.
- You must write your final worker report to: {result_path}
- If you create extra intermediate artifacts, keep them inside: {artifacts_dir}
- Your final worker report should be a concise Markdown document that the lead agent can later read and synthesize.
</worker_contract>

<runtime_context>
{system_context}
</runtime_context>
"""


class SubAgentRunner:
    def __init__(self, llm, tools: list[BaseTool], max_steps: int = 4, system_context: str = "") -> None:
        self._llm = llm.bind_tools(tools)
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._valid_tool_names = set(self._tools_by_name)
        self._max_steps = max_steps
        self._system_context = system_context.strip()

    def run(self, task_path: Path, result_path: Path, artifacts_dir: Path, system_context: str = "") -> str:
        task_text = task_path.read_text(encoding="utf-8").strip()
        system_prompt = SUBAGENT_SYSTEM_PROMPT.format(
            system_context=(system_context.strip() or self._system_context).strip(),
            result_path=result_path,
            artifacts_dir=artifacts_dir,
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=task_text)]
        for _ in range(self._max_steps):
            ai_msg = self._llm.invoke(messages)
            ai_msg = coerce_json_tool_call(ai_msg, self._valid_tool_names)
            messages.append(ai_msg)
            if not ai_msg.tool_calls:
                break

            for tool_call in ai_msg.tool_calls:
                name = tool_call["name"]
                tool = self._tools_by_name.get(name)
                if tool is None:
                    tool_result = f"Tool not found: {name}"
                else:
                    try:
                        tool_result = tool.invoke(tool_call.get("args", {}))
                    except Exception as exc:  # noqa: BLE001
                        tool_result = f"Tool execution failed for {name}: {exc}"
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=name,
                        tool_call_id=tool_call["id"],
                    )
                )
        if not result_path.exists() or not result_path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"Subagent did not write a final result file: {result_path}")
        return f"Subagent completed and wrote result to {result_path}"
