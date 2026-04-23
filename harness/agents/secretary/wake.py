from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from harness.agents.human_loop import HumanLoopManager
from harness.agents.memory.manager import ConversationMemoryManager
from harness.agents.memory_postprocess import MemoryPostprocessQueue
from harness.agents.secretary.mind import SECRETARY_MIND_TEMPLATE
from harness.agents.secretary.state import FalcoState
from harness.agents.thread_session import ThreadSessionManager
from harness.agents.subagent import SubAgentRunner
from harness.agents.subagent_tasks import SubAgentTaskManager
from harness.agents.tool_calling import coerce_json_tool_call
from harness.config.config import FalcoSettings
from harness.mcp import MCPToolRegistry
from harness.skills.skills import SkillManager
from harness.tools.tools import create_core_tools, create_subagent_tools, execute_pending_approval
from harness.workspace import WorkspaceManager

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


class FalcoOrchestrator:
    def __init__(self, settings: FalcoSettings | None = None) -> None:
        load_dotenv()
        self.settings = settings or FalcoSettings.from_yaml(Path(__file__).resolve().parents[3] / "config.yaml")

        self.workspace = WorkspaceManager(
            primary_root=self.settings.workspace_root,
            allowed_roots=self.settings.allowed_workspace_roots,
            default_working_directory=self.settings.default_working_directory,
            uploads_root=self.settings.uploads_root,
            runtime_root=self.settings.runtime_root,
            deliverables_root=self.settings.deliverables_root,
            thread_scoped_directories=self.settings.thread_scoped_directories,
        )

        self._thread_id_ctx: ContextVar[str] = ContextVar("falco_thread_id", default="default")
        self._latest_user_ctx: ContextVar[str] = ContextVar("falco_latest_user", default="")
        self._response_preference_ctx: ContextVar[str] = ContextVar("falco_response_preference", default="natural")
        self._resume_input_ctx: ContextVar[str] = ContextVar("falco_resume_input", default="")
        self._working_directory_ctx: ContextVar[str] = ContextVar(
            "falco_working_directory",
            default=str(self.workspace.default_working_directory),
        )
        self._thread_workdirs: dict[str, str] = {}

        self.memory = ConversationMemoryManager(
            self.settings.memory_root,
            recent_rounds=self.settings.memory_recent_rounds,
            key_rounds=self.settings.memory_key_rounds,
            importance_threshold=self.settings.memory_importance_threshold,
            max_rounds=self.settings.memory_max_rounds,
            daily_half_life_days=self.settings.memory_daily_half_life_days,
            daily_lookback_days=self.settings.memory_daily_lookback_days,
            daily_retrieval_items=self.settings.memory_daily_retrieval_items,
            evergreen_retrieval_items=self.settings.memory_evergreen_retrieval_items,
            tokenizer_model=self.settings.model,
        )
        self.memory_postprocess = MemoryPostprocessQueue()
        self.human_loop = HumanLoopManager(self.settings.workspace_root / ".falco" / "hitl")
        self.thread_sessions = ThreadSessionManager(self.settings.workspace_root / ".falco" / "threads")
        self.subagent_tasks = SubAgentTaskManager()
        self.skills = SkillManager(
            public_root=self.settings.skills_public_root,
            user_roots=self.settings.skills_user_roots,
        )
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
        self.mcp_tools = self.mcp.load_tools()
        self.graph = self._build_graph()

    def _build_graph(self):
        def run_subagent_tasks_for_thread(thread_id: str, worker_ids: list[str] | None = None) -> list[dict[str, str]]:
            runtime_dir = self.workspace.thread_directory(thread_id, "runtime")
            items = self.subagent_tasks.list_tasks(runtime_dir=runtime_dir)
            selected_ids = {item.strip() for item in (worker_ids or []) if str(item).strip()}
            if selected_ids:
                items = [item for item in items if item.get("worker_id") in selected_ids]
            items = [item for item in items if item.get("status") == "pending"]
            reports: list[dict[str, str]] = []
            for item in items:
                worker_id = str(item["worker_id"])
                worker_root = Path(item["worker_root"])
                task_path = Path(item["task_path"])
                result_path = Path(item["result_path"])
                artifacts_dir = Path(item["artifacts_dir"])
                self.subagent_tasks.mark_running(runtime_dir=runtime_dir, worker_id=worker_id)
                runner = SubAgentRunner(
                    llm=self.llm,
                    tools=create_subagent_tools(
                        workspace=self.workspace,
                        settings=self.settings,
                        thread_id=thread_id,
                        worker_root=worker_root,
                    ),
                    max_steps=self.settings.subagent_max_steps,
                    system_context="Follow the worker file contract exactly.",
                )
                try:
                    message = runner.run(
                        task_path=task_path,
                        result_path=result_path,
                        artifacts_dir=artifacts_dir,
                        system_context=self.workspace.prompt_block(thread_id=thread_id, cwd=worker_root),
                    )
                    self.subagent_tasks.mark_completed(runtime_dir=runtime_dir, worker_id=worker_id)
                    reports.append(
                        {
                            "worker_id": worker_id,
                            "status": "completed",
                            "result_path": self.workspace.describe_path(result_path, thread_id=thread_id),
                            "message": message,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        f"# Worker Failure\n\nThe worker could not complete its task.\n\nReason: {exc}\n",
                        encoding="utf-8",
                    )
                    self.subagent_tasks.mark_failed(runtime_dir=runtime_dir, worker_id=worker_id, reason=str(exc))
                    reports.append(
                        {
                            "worker_id": worker_id,
                            "status": "failed",
                            "result_path": self.workspace.describe_path(result_path, thread_id=thread_id),
                            "message": str(exc),
                        }
                    )
            return reports

        tools = create_core_tools(
            workspace=self.workspace,
            memory=self.memory,
            human_loop=self.human_loop,
            skills=self.skills,
            settings=self.settings,
            llm=self.llm,
            thread_id_getter=self._thread_id_ctx.get,
            latest_user_getter=self._latest_user_ctx.get,
            working_directory_getter=self._working_directory_ctx.get,
            working_directory_setter=self._working_directory_ctx.set,
            working_directory_persistor=self.thread_sessions.set_working_directory,
            resume_input_getter=self._resume_input_ctx.get,
            mcp_catalog_getter=self.mcp.catalog,
            mcp_reload_getter=self._reload_mcp_tools_now,
            subagent_task_manager=self.subagent_tasks,
            subagent_task_runner=run_subagent_tasks_for_thread,
            max_subagents=self.settings.max_subagents,
            include_delegate=True,
        ) + self.mcp_tools

        model_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)
        valid_tool_names = {tool.name for tool in tools}

        def build_tools_block() -> str:
            lines = ["<tools>"]
            for tool in tools:
                description = " ".join(str(tool.description or "").split())
                lines.append(f"- {tool.name}: {description}")
            lines.append("</tools>")
            return "\n".join(lines)

        def hydrate_context(state: FalcoState) -> FalcoState:
            tid = state.get("thread_id") or self._thread_id_ctx.get()
            self._thread_id_ctx.set(tid)
            self.workspace.ensure_thread_directories(tid)

            working_directory = self.workspace.resolve_working_directory(
                state.get("working_directory") or self._working_directory_ctx.get(),
                thread_id=tid,
            )
            self._working_directory_ctx.set(str(working_directory))
            self._thread_workdirs[tid] = str(working_directory)
            self.thread_sessions.set_working_directory(tid, str(working_directory))

            latest_user_text = ""
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    latest_user_text = str(message.content or "")
                    break
            self._latest_user_ctx.set(latest_user_text)
            self._response_preference_ctx.set(str(state.get("user_response_preference") or self._response_preference_ctx.get()))

            return {
                "thread_id": tid,
                "user_response_preference": self._response_preference_ctx.get(),
                "context_block": self.memory.build_context_block(
                    tid,
                    max_items=self.settings.max_context_messages,
                    recent_rounds=self.settings.memory_recent_rounds,
                    key_rounds=self.settings.memory_key_rounds,
                    query_hint=latest_user_text,
                    max_tokens=self.settings.memory_context_max_tokens,
                ),
                "soul_block": self._load_soul_block(),
                "skills_block": self.skills.get_prompt_block(),
                "mcp_block": self.mcp.prompt_block(),
                "workspace_block": self.workspace.prompt_block(thread_id=tid, cwd=working_directory),
                "working_directory": str(working_directory),
            }

        def lead_agent(state: FalcoState) -> FalcoState:
            context_block = state.get("context_block", "").strip()
            skills_block = state.get("skills_block", "").strip()
            dynamic_prompt = SECRETARY_MIND_TEMPLATE.format(
                soul=state.get("soul_block", "").strip(),
                memory=context_block,
                user_response_preference=self._resolve_user_response_preference(
                    str(state.get("user_response_preference") or self._response_preference_ctx.get()),
                ),
                working_environment=state.get("workspace_block", "").strip(),
                skils=skills_block,
                mcp=state.get("mcp_block", "").strip(),
                tools=build_tools_block(),
            )
            conversation = [SystemMessage(content=dynamic_prompt), *state["messages"]]
            tool_message_count = sum(1 for message in state["messages"] if isinstance(message, ToolMessage))
            tool_budget_reached = tool_message_count >= max(1, self.settings.max_tool_steps - 1)
            if tool_budget_reached:
                forced_prompt = (
                    dynamic_prompt
                    + "\n\n<tool_budget>\n"
                    + "You have reached the tool-call budget for this turn. "
                    + "Do not call any more tools. "
                    + "Use the tool results already present in the conversation to produce the best possible final answer."
                    + "\n</tool_budget>"
                )
                response = self.llm.invoke([SystemMessage(content=forced_prompt), *state["messages"]])
            else:
                response = model_with_tools.invoke(conversation)
            if isinstance(response, AIMessage) and not tool_budget_reached:
                response = coerce_json_tool_call(response, valid_tool_names)
            return {"messages": [response]}

        def route_next(state: FalcoState) -> Literal["tools", "persist"]:
            last = state["messages"][-1]
            tool_message_count = sum(1 for message in state["messages"] if isinstance(message, ToolMessage))
            if tool_message_count >= max(1, self.settings.max_tool_steps - 1):
                return "persist"
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return "persist"

        def handle_hitl(state: FalcoState) -> FalcoState:
            last = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {}
            payload = self._extract_hitl_payload(
                thread_id=state.get("thread_id") or self._thread_id_ctx.get(),
                tool_message=str(last.content or ""),
            )
            if payload is None:
                return {}
            resume_value = interrupt(payload)
            return self._apply_hitl_resume(
                thread_id=state.get("thread_id") or self._thread_id_ctx.get(),
                payload=payload,
                resume_value=resume_value,
            )

        def persist_memory(state: FalcoState) -> FalcoState:
            tid = state.get("thread_id") or self._thread_id_ctx.get()
            human_messages: list[str] = []
            latest_ai = ""
            tool_observations: list[str] = []
            runtime_context_tokens = 0
            for message in state.get("messages", []):
                content = str(getattr(message, "content", "") or "")
                runtime_context_tokens += self.memory._count_tokens(content)
                if isinstance(message, HumanMessage) and content.strip():
                    human_messages.append(content.strip())
            runtime_context_tokens += self.memory._count_tokens(state.get("context_block", ""))
            for message in reversed(state["messages"]):
                if isinstance(message, ToolMessage) and len(tool_observations) < self.settings.max_tool_steps:
                    tool_observations.append(str(message.content or ""))
                if not latest_ai and isinstance(message, AIMessage) and str(message.content or "").strip():
                    latest_ai = str(message.content or "")
                if latest_ai:
                    continue
            latest_user = self._merge_human_messages_for_memory(human_messages)
            if not latest_user.strip() and not latest_ai.strip():
                return {}

            reversed_tool_observations = list(reversed(tool_observations))

            def run_memory_postprocess() -> None:
                self.memory.add_round(
                    tid,
                    user=latest_user,
                    assistant=latest_ai,
                    llm=self.llm,
                )
                self.memory.maybe_run_silent_turn_compaction(
                    thread_id=tid,
                    llm=self.llm,
                    context_soft_limit_tokens=self.settings.memory_context_soft_limit_tokens,
                    context_max_tokens=self.settings.memory_context_max_tokens,
                    silent_turn_cooldown_rounds=self.settings.memory_silent_turn_cooldown_rounds,
                    query_hint=latest_user,
                    runtime_context_tokens=runtime_context_tokens,
                )
                self.memory.reflect_on_turn(
                    thread_id=tid,
                    user=latest_user,
                    assistant=latest_ai,
                    tool_observations=reversed_tool_observations,
                    llm=self.llm,
                )

            self.memory_postprocess.enqueue(thread_id=tid, fn=run_memory_postprocess)
            return {}

        builder = StateGraph(FalcoState)
        builder.add_node("hydrate_context", hydrate_context)
        builder.add_node("lead_agent", lead_agent)
        builder.add_node("tools", tool_node)
        builder.add_node("handle_hitl", handle_hitl)
        builder.add_node("persist", persist_memory)
        builder.add_edge(START, "hydrate_context")
        builder.add_edge("hydrate_context", "lead_agent")
        builder.add_conditional_edges("lead_agent", route_next, ["tools", "persist"])
        builder.add_edge("tools", "handle_hitl")
        builder.add_edge("handle_hitl", "lead_agent")
        builder.add_edge("persist", END)

        return builder.compile(checkpointer=InMemorySaver())

    def invoke(self, user_input: str, thread_id: str = "default", user_response_preference: str = "natural") -> str:
        self.memory_postprocess.flush_thread(thread_id)
        self._refresh_mcp_runtime_if_needed()
        self._thread_id_ctx.set(thread_id)
        self._latest_user_ctx.set(user_input)
        self._response_preference_ctx.set(user_response_preference or "natural")
        self._resume_input_ctx.set("")
        working_directory = self._restore_thread_working_directory(thread_id)
        self._working_directory_ctx.set(working_directory)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(12, self.settings.max_tool_steps * 3 + 8),
        }
        state: FalcoState = {
            "messages": [HumanMessage(content=user_input)],
            "thread_id": thread_id,
            "user_response_preference": user_response_preference or "natural",
            "working_directory": working_directory,
        }
        result = self.graph.invoke(state, config=config)
        if isinstance(result, dict) and result.get("working_directory"):
            resolved = self.workspace.resolve_working_directory(str(result["working_directory"]), thread_id=thread_id)
            self._thread_workdirs[thread_id] = str(resolved)
            self.thread_sessions.set_working_directory(thread_id, str(resolved))
        interrupt_text = self._render_interrupt_result(result)
        if interrupt_text:
            return interrupt_text
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and str(message.content or "").strip():
                return str(message.content)
            if isinstance(message, ToolMessage) and str(message.content or "").strip():
                return str(message.content)
        return ""

    def resume(self, user_input: str, thread_id: str = "default", user_response_preference: str = "natural") -> str:
        self.memory_postprocess.flush_thread(thread_id)
        self._refresh_mcp_runtime_if_needed()
        self._thread_id_ctx.set(thread_id)
        self._latest_user_ctx.set(user_input)
        self._response_preference_ctx.set(user_response_preference or "natural")
        self._resume_input_ctx.set(user_input)
        self._working_directory_ctx.set(self._restore_thread_working_directory(thread_id))
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(12, self.settings.max_tool_steps * 3 + 8),
        }
        result = self.graph.invoke(Command(resume=user_input), config=config)
        if isinstance(result, dict) and result.get("working_directory"):
            resolved = self.workspace.resolve_working_directory(str(result["working_directory"]), thread_id=thread_id)
            self._thread_workdirs[thread_id] = str(resolved)
            self.thread_sessions.set_working_directory(thread_id, str(resolved))
        interrupt_text = self._render_interrupt_result(result)
        if interrupt_text:
            return interrupt_text
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and str(message.content or "").strip():
                return str(message.content)
            if isinstance(message, ToolMessage) and str(message.content or "").strip():
                return str(message.content)
        return ""

    def _render_interrupt_result(self, result) -> str:
        if not isinstance(result, dict):
            return ""
        interrupts = result.get("__interrupt__") or []
        if not interrupts:
            return ""
        payload = getattr(interrupts[0], "value", interrupts[0])
        if not isinstance(payload, dict):
            return str(payload)

        marker = "HUMAN_APPROVAL_REQUIRED" if payload.get("kind") == "approval" else "HUMAN_INPUT_REQUIRED"
        lines = [marker]
        if payload.get("request_id"):
            lines.append(f"id={payload['request_id']}")
        if payload.get("clarification_type"):
            lines.append(f"clarification_type={payload['clarification_type']}")
        if payload.get("action"):
            lines.append(f"action={payload['action']}")
        if payload.get("question"):
            lines.append(f"question={payload['question']}")
        if payload.get("rationale"):
            lines.append(f"rationale={payload['rationale']}")
        options = payload.get("options") or []
        if options:
            lines.append("options=" + " | ".join(str(item) for item in options))
        lines.append("Stop and wait for user response.")
        return "\n".join(lines)

    def _extract_hitl_payload(self, *, thread_id: str, tool_message: str) -> dict | None:
        text = tool_message.strip()
        if not text:
            return None
        if "HUMAN_INPUT_REQUIRED" not in text and "HUMAN_APPROVAL_REQUIRED" not in text:
            return None
        fields: dict[str, str] = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
        request_id = fields.get("id", "")
        pending = self.human_loop.get_pending(thread_id, request_id) if request_id else None
        if text.startswith("HUMAN_APPROVAL_REQUIRED"):
            return {
                "kind": "approval",
                "request_id": request_id,
                "action": fields.get("action", pending.get("action", "") if pending else ""),
                "question": fields.get("question", pending.get("question", "") if pending else ""),
                "rationale": fields.get("rationale", pending.get("rationale", "") if pending else ""),
                "preview": fields.get("preview", ""),
            }
        return {
            "kind": "clarification",
            "request_id": request_id,
            "question": fields.get("question", pending.get("question", "") if pending else ""),
            "clarification_type": fields.get(
                "clarification_type",
                pending.get("clarification_type", "missing_info") if pending else "missing_info",
            ),
            "context": pending.get("context", "") if pending else "",
            "options": pending.get("options", []) if pending else [],
        }

    def _apply_hitl_resume(self, *, thread_id: str, payload: dict, resume_value) -> FalcoState:
        request_id = str(payload.get("request_id", "")).strip()
        if payload.get("kind") == "clarification":
            answer = str(resume_value or "").strip()
            if request_id:
                self.human_loop.mark_completed(thread_id, request_id, f"User answered: {answer}")
            return {"messages": [HumanMessage(content=answer)]}

        approved = self._is_resume_approved(resume_value)
        if not approved:
            if request_id:
                self.human_loop.mark_denied(thread_id, request_id, str(resume_value or "").strip())
            return {"messages": [HumanMessage(content=str(resume_value or "").strip() or "denied")]}

        try:
            result = execute_pending_approval(
                thread_id=thread_id,
                request_id=request_id,
                human_loop=self.human_loop,
                workspace=self.workspace,
                current_working_directory=self.workspace.resolve_working_directory(
                    self._working_directory_ctx.get(),
                    thread_id=thread_id,
                ),
                skills=self.skills,
                settings=self.settings,
                llm=self.llm,
            )
        except Exception as exc:  # noqa: BLE001
            result = f"Approval execution failed for {request_id or 'pending_request'}: {exc}"
        return {
            "messages": [
                HumanMessage(content=str(resume_value or "").strip() or "approved"),
                ToolMessage(
                    content=result,
                    name="approve_pending_action",
                    tool_call_id=request_id or "resume_approval",
                ),
            ]
        }

    def _is_resume_approved(self, value) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"y", "yes", "true", "approve", "approved", "confirm", "ok", "okay"}

    def _restore_thread_working_directory(self, thread_id: str) -> str:
        saved = self.thread_sessions.get_working_directory(thread_id)
        self.workspace.ensure_thread_directories(thread_id)
        if saved:
            try:
                resolved = self.workspace.resolve_working_directory(saved, thread_id=thread_id)
                self._thread_workdirs[thread_id] = str(resolved)
                return str(resolved)
            except ValueError:
                pass
        fallback = str(self.workspace.default_working_directory)
        self._thread_workdirs[thread_id] = fallback
        self.thread_sessions.set_working_directory(thread_id, fallback)
        return fallback

    def _merge_human_messages_for_memory(self, human_messages: list[str]) -> str:
        items = [item.strip() for item in human_messages if item.strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        merged = [items[0], "", "Clarification responses:"]
        for item in items[1:]:
            merged.append(f"- {item}")
        return "\n".join(merged).strip()

    def _load_soul_block(self) -> str:
        soul_path = self.settings.soul_path
        if soul_path is None or not soul_path.exists() or not soul_path.is_file():
            return ""
        try:
            content = soul_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
        if not content:
            return ""
        return f"<soul>\n{content}\n</soul>"

    def _resolve_user_response_preference(self, preference: str) -> str:
        mapping = {
            "natural": "- Natural Tone: Use clear paragraphs and natural prose by default\n- Action-Oriented: Focus on delivering useful results, not explaining internal process",
            "concise": "- Concise: Keep replies short and direct\n- High Signal: Prioritize conclusions, decisions, and next actions over background detail",
            "professional": "- Professional: Use structured, precise, businesslike language\n- Reliable: Emphasize clarity, correctness, and explicit recommendations",
            "warm": "- Warm: Sound supportive, friendly, and collaborative\n- Clear: Stay approachable while still being practical and decisive",
            "direct": "- Direct: Get to the point quickly and avoid unnecessary framing\n- Decisive: Prefer firm recommendations and explicit answers",
        }
        return mapping.get(preference, mapping["natural"])

    def _refresh_mcp_runtime_if_needed(self) -> None:
        if self.mcp.reload_if_needed():
            self.mcp_tools = self.mcp.tools()
            self.graph = self._build_graph()

    def _reload_mcp_tools_now(self) -> str:
        self.mcp.reload_if_needed(force=True)
        self.mcp_tools = self.mcp.tools()
        self.graph = self._build_graph()
        return self.mcp.catalog()

    def reload_mcp_tools(self) -> str:
        return self._reload_mcp_tools_now()

    def mcp_catalog(self) -> str:
        self._refresh_mcp_runtime_if_needed()
        return self.mcp.catalog()
