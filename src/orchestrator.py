from __future__ import annotations

from contextvars import ContextVar
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from config import FalcoSettings
from memory import ConversationMemoryManager
from rag import MilvusRAG
from skills import SkillManager
from state import FalcoState
from subagent import SubAgentRunner
from tools import create_core_tools


LEAD_SYSTEM_PROMPT = """You are Falco lead agent.
You orchestrate tools and delegated sub-agents to solve user tasks.

Operating protocol:
1) Plan minimally, act quickly, validate outputs.
2) Use tools for filesystem, memory, and skills.
3) Delegate independent or deep subtasks via `delegate_task`.
4) Use `rag_search` when local knowledge-base evidence is needed.
5) Keep responses concise, concrete, and execution-oriented.
"""


class FalcoOrchestrator:
    def __init__(self, settings: FalcoSettings | None = None) -> None:
        load_dotenv()
        self.settings = settings or FalcoSettings.from_env()
        self._thread_id_ctx: ContextVar[str] = ContextVar("falco_thread_id", default="default")
        self.memory = ConversationMemoryManager(
            self.settings.memory_root,
            recent_rounds=self.settings.memory_recent_rounds,
            key_rounds=self.settings.memory_key_rounds,
            importance_threshold=self.settings.memory_importance_threshold,
            max_rounds=self.settings.memory_max_rounds,
        )
        self.skills = SkillManager(self.settings.skills_root)
        self.llm = ChatOpenAI(
            model=self.settings.model,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            temperature=0,
        )
        self.rag = MilvusRAG(self.settings, llm=self.llm) if self.settings.rag_enabled else None
        self.graph = self._build_graph()

    def _build_graph(self):
        base_tools = create_core_tools(
            workspace_root=self.settings.workspace_root,
            memory=self.memory,
            skills=self.skills,
            rag=self.rag,
            thread_id_getter=self._thread_id_ctx.get,
            include_delegate=False,
        )
        subagent_runner = SubAgentRunner(
            llm=self.llm,
            tools=base_tools,
            max_steps=max(2, self.settings.max_tool_steps - 1),
        )
        tools = create_core_tools(
            workspace_root=self.settings.workspace_root,
            memory=self.memory,
            skills=self.skills,
            rag=self.rag,
            thread_id_getter=self._thread_id_ctx.get,
            subagent_runner=subagent_runner,
            include_delegate=True,
        )

        model_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        def hydrate_context(state: FalcoState) -> FalcoState:
            tid = state.get("thread_id") or self._thread_id_ctx.get()
            self._thread_id_ctx.set(tid)
            return {
                "thread_id": tid,
                "context_block": self.memory.build_context_block(
                    tid,
                    max_items=self.settings.max_context_messages,
                    recent_rounds=self.settings.memory_recent_rounds,
                    key_rounds=self.settings.memory_key_rounds,
                ),
                "skills_block": self.skills.get_prompt_block(),
            }

        def lead_agent(state: FalcoState) -> FalcoState:
            context_block = state.get("context_block", "").strip() or "No prior memory context."
            skills_block = state.get("skills_block", "").strip() or "No active skills."
            dynamic_prompt = (
                f"{LEAD_SYSTEM_PROMPT}\n\n"
                f"Memory context:\n{context_block}\n\n"
                f"Active skills:\n{skills_block}\n"
            )
            response = model_with_tools.invoke(
                [SystemMessage(content=dynamic_prompt), *state["messages"]]
            )
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
            for message in reversed(state["messages"]):
                if not latest_ai and isinstance(message, AIMessage):
                    latest_ai = str(message.content or "")
                if not latest_user and isinstance(message, HumanMessage):
                    latest_user = str(message.content or "")
                if latest_user and latest_ai:
                    break
            self.memory.add_round(
                tid,
                user=latest_user,
                assistant=latest_ai,
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
        config = {"configurable": {"thread_id": thread_id}}
        state: FalcoState = {"messages": [HumanMessage(content=user_input)], "thread_id": thread_id}
        result = self.graph.invoke(state, config=config)
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                return str(message.content)
        return ""
