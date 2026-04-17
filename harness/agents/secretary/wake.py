from __future__ import annotations

from contextvars import ContextVar
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from harness.agents.human_loop import HumanLoopManager
from harness.agents.memory.manager import ConversationMemoryManager
from harness.agents.secretary.state import FalcoState
from harness.agents.subagent import SubAgentRunner
from harness.agents.tool_calling import coerce_json_tool_call
from harness.config.config import FalcoSettings
from harness.mcp import MCPToolRegistry
from harness.prompts.templates import LEAD_REACT_PROMPT_TEMPLATE
from harness.rag import MilvusRAG
from harness.skills.skills import SkillManager
from harness.tools.tools import create_core_tools

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


LEAD_SYSTEM_PROMPT = (
    LEAD_REACT_PROMPT_TEMPLATE
    + """
Operating protocol:
1) Think privately about the next smallest useful action; do not expose hidden reasoning.
2) Act by calling tools when external state, memory, files, skills, or delegated work are needed.
3) Observe tool results carefully before deciding the next action.
4) Delegate independent or deep subtasks via `delegate_task`.
5) Use `use_skill(skill_name="rag", action="search", args={...})` when local knowledge-base evidence is needed.
6) Use `mcp_catalog` to inspect external MCP servers when external capabilities may be relevant.
7) Use MCP tools only when their names and descriptions match the task.
8) Use `request_user_input` when required information is missing.
9) When the user answers a pending clarification, call `record_user_input` with its id before continuing.
10) Mutating operations may return HUMAN_APPROVAL_REQUIRED; stop and ask the user to approve or deny the id.
11) Answer only after enough observation or when no tool is needed; keep responses concise and concrete.
"""
)


class FalcoOrchestrator:
    def __init__(self, settings: FalcoSettings | None = None) -> None:
        load_dotenv()
        self.settings = settings or FalcoSettings.from_env()
        self._thread_id_ctx: ContextVar[str] = ContextVar("falco_thread_id", default="default")
        self._latest_user_ctx: ContextVar[str] = ContextVar("falco_latest_user", default="")
        self.memory = ConversationMemoryManager(
            self.settings.memory_root,
            recent_rounds=self.settings.memory_recent_rounds,
            key_rounds=self.settings.memory_key_rounds,
            importance_threshold=self.settings.memory_importance_threshold,
            max_rounds=self.settings.memory_max_rounds,
        )
        self.human_loop = HumanLoopManager(self.settings.workspace_root / ".falco" / "hitl")
        self.skills = SkillManager(self.settings.skills_root)
        self.mcp = MCPToolRegistry(
            config_path=self.settings.mcp_config_path,
            enabled=self.settings.mcp_enabled,
            prefix_tools=self.settings.mcp_tool_prefix,
        )
        self.llm = ChatOpenAI(
            model=self.settings.model,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            temperature=0,
        )
        self.rag = MilvusRAG(self.settings, llm=self.llm) if self.settings.rag_enabled else None
        self.mcp_tools = self.mcp.load_tools()
        self.graph = self._build_graph()

    def _build_graph(self):
        base_tools = create_core_tools(
            workspace_root=self.settings.workspace_root,
            memory=self.memory,
            human_loop=self.human_loop,
            skills=self.skills,
            rag=self.rag,
            thread_id_getter=self._thread_id_ctx.get,
            latest_user_getter=self._latest_user_ctx.get,
            mcp_catalog_getter=self.mcp.catalog,
            include_delegate=False,
        ) + self.mcp_tools
        subagent_runner = SubAgentRunner(
            llm=self.llm,
            tools=base_tools,
            max_steps=max(2, self.settings.max_tool_steps - 1),
        )
        tools = create_core_tools(
            workspace_root=self.settings.workspace_root,
            memory=self.memory,
            human_loop=self.human_loop,
            skills=self.skills,
            rag=self.rag,
            thread_id_getter=self._thread_id_ctx.get,
            latest_user_getter=self._latest_user_ctx.get,
            mcp_catalog_getter=self.mcp.catalog,
            subagent_runner=subagent_runner,
            include_delegate=True,
        ) + self.mcp_tools

        model_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)
        valid_tool_names = {tool.name for tool in tools}

        def hydrate_context(state: FalcoState) -> FalcoState:
            tid = state.get("thread_id") or self._thread_id_ctx.get()
            self._thread_id_ctx.set(tid)
            latest_user_text = ""
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    latest_user_text = str(message.content or "")
                    break
            self._latest_user_ctx.set(latest_user_text)
            return {
                "thread_id": tid,
                "context_block": self.memory.build_context_block(
                    tid,
                    max_items=self.settings.max_context_messages,
                    recent_rounds=self.settings.memory_recent_rounds,
                    key_rounds=self.settings.memory_key_rounds,
                    query_hint=latest_user_text,
                    max_chars=self.settings.memory_context_max_chars,
                ),
                "skills_block": self.skills.get_prompt_block(),
            }

        def lead_agent(state: FalcoState) -> FalcoState:
            context_block = state.get("context_block", "").strip()
            skills_block = state.get("skills_block", "").strip()
            dynamic_prompt = (
                f"{LEAD_SYSTEM_PROMPT}\n\n"
                f"Memory context:\n{context_block}\n\n"
                f"Active skills:\n{skills_block}\n"
            )
            response = model_with_tools.invoke(
                [SystemMessage(content=dynamic_prompt), *state["messages"]]
            )
            if isinstance(response, AIMessage):
                response = coerce_json_tool_call(response, valid_tool_names)
            return {"messages": [response]}

        def route_next(state: FalcoState) -> Literal["tools", "persist"]:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return "persist"

        def persist_memory(state: FalcoState) -> FalcoState:
            tid = state.get("thread_id") or self._thread_id_ctx.get()
            latest_user = ""
            latest_ai = ""
            tool_observations: list[str] = []
            for message in reversed(state["messages"]):
                if isinstance(message, ToolMessage) and len(tool_observations) < self.settings.max_tool_steps:
                    tool_observations.append(str(message.content or ""))
                if not latest_ai and isinstance(message, AIMessage) and str(message.content or "").strip():
                    latest_ai = str(message.content or "")
                if not latest_user and isinstance(message, HumanMessage):
                    latest_user = str(message.content or "")
                if latest_user and latest_ai:
                    continue
            self.memory.add_round(
                tid,
                user=latest_user,
                assistant=latest_ai,
                llm=self.llm,
            )
            self.memory.maybe_run_silent_turn_compaction(
                thread_id=tid,
                llm=self.llm,
                context_soft_limit_chars=self.settings.memory_context_soft_limit_chars,
                context_max_chars=self.settings.memory_context_max_chars,
                silent_turn_cooldown_rounds=self.settings.memory_silent_turn_cooldown_rounds,
                query_hint=latest_user,
            )
            self.memory.reflect_on_turn(
                thread_id=tid,
                user=latest_user,
                assistant=latest_ai,
                tool_observations=list(reversed(tool_observations)),
                llm=self.llm,
            )
            return {}

        builder = StateGraph(FalcoState)
        builder.add_node("hydrate_context", hydrate_context)
        builder.add_node("lead_agent", lead_agent)
        builder.add_node("tools", tool_node)
        builder.add_node("persist", persist_memory)
        builder.add_edge(START, "hydrate_context")
        builder.add_edge("hydrate_context", "lead_agent")
        builder.add_conditional_edges("lead_agent", route_next, ["tools", "persist"])
        builder.add_edge("tools", "lead_agent")
        builder.add_edge("persist", END)

        return builder.compile(checkpointer=InMemorySaver())

    def invoke(self, user_input: str, thread_id: str = "default") -> str:
        self._thread_id_ctx.set(thread_id)
        self._latest_user_ctx.set(user_input)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(6, self.settings.max_tool_steps * 2 + 4),
        }
        state: FalcoState = {"messages": [HumanMessage(content=user_input)], "thread_id": thread_id}
        result = self.graph.invoke(state, config=config)
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and str(message.content or "").strip():
                return str(message.content)
        return ""
