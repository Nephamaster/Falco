from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from harness.prompts.templates import SUBAGENT_PROMPT_TEMPLATE
from harness.agents.tool_calling import coerce_json_tool_call


SUBAGENT_SYSTEM_PROMPT = SUBAGENT_PROMPT_TEMPLATE


class SubAgentRunner:
    def __init__(self, llm, tools: list[BaseTool], max_steps: int = 4) -> None:
        self._llm = llm.bind_tools(tools)
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._valid_tool_names = set(self._tools_by_name)
        self._max_steps = max_steps

    def run(self, task: str, context: str = "") -> str:
        user_request = f"Task:\n{task.strip()}"
        if context.strip():
            user_request += f"\n\nContext:\n{context.strip()}"

        messages = [SystemMessage(content=SUBAGENT_SYSTEM_PROMPT), HumanMessage(content=user_request)]
        final_text = ""
        for _ in range(self._max_steps):
            ai_msg = self._llm.invoke(messages)
            ai_msg = coerce_json_tool_call(ai_msg, self._valid_tool_names)
            messages.append(ai_msg)
            if ai_msg.content:
                final_text = str(ai_msg.content)
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

        return final_text.strip() or "Subagent finished without textual output."
