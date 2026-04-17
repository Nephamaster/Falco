from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from harness.agents.human_loop import HumanLoopManager
from harness.agents.memory.manager import ConversationMemoryManager
from harness.agents.subagent import SubAgentRunner
from harness.rag import MilvusRAG
from harness.skills.skills import SkillManager

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


load_dotenv()

MAX_TOOL_READ_CHARS = 100_000
MAX_TOOL_WRITE_CHARS = 200_000
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_SEARCH_FILES = 800
MAX_LIST_ITEMS = 300

BLOCKED_PATH_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".falco",
    "__pycache__",
    "node_modules",
    ".next",
    ".venv",
    "venv",
}

BLOCKED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "id_rsa",
    "id_ed25519",
}

BLOCKED_FILE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}


def _resolve_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = (workspace_root / raw_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:  # noqa: B904
        raise ValueError("Path escapes workspace root.") from exc
    return candidate


def _is_blocked_path(path: Path, workspace_root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return True
    parts = set(rel.parts)
    if parts & BLOCKED_PATH_PARTS:
        return True
    if path.name in BLOCKED_FILE_NAMES:
        return True
    return path.suffix.lower() in BLOCKED_FILE_SUFFIXES


def _ensure_tool_path_allowed(path: Path, workspace_root: Path, *, write: bool = False) -> None:
    if _is_blocked_path(path, workspace_root):
        action = "write" if write else "access"
        try:
            rel = path.resolve().relative_to(workspace_root.resolve())
        except ValueError:
            rel = path
        raise ValueError(f"Refusing to {action} sensitive or generated path: {rel}")


def _iter_search_files(root: Path, workspace_root: Path) -> list[Path]:
    if root.is_file():
        return [root] if not _is_blocked_path(root, workspace_root) else []

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_blocked_path(current / name, workspace_root)
        ]
        for name in filenames:
            file_path = current / name
            if _is_blocked_path(file_path, workspace_root):
                continue
            files.append(file_path)
            if len(files) >= MAX_SEARCH_FILES:
                return files
    return files


def create_core_tools(
    *,
    workspace_root: Path,
    memory: ConversationMemoryManager,
    human_loop: HumanLoopManager,
    skills: SkillManager,
    rag: MilvusRAG | None,
    thread_id_getter,
    latest_user_getter,
    mcp_catalog_getter=None,
    subagent_runner: SubAgentRunner | None = None,
    include_delegate: bool = True,
) -> list[BaseTool]:
    def _approval_required(item: dict, preview: str = "") -> str:
        lines = [
            "HUMAN_APPROVAL_REQUIRED",
            f"id={item['id']}",
            f"action={item.get('action', '')}",
            f"rationale={item.get('rationale', '')}",
        ]
        if preview:
            lines.append(f"preview={preview}")
        lines.append("Ask the user to approve or deny this id before continuing.")
        return "\n".join(lines)

    def _latest_user_approves(request_id: str) -> bool:
        latest_user = latest_user_getter().lower()
        approval_words = ("approve", "approved", "yes", "confirm", "批准", "同意", "确认", "可以")
        return request_id.lower() in latest_user and any(word in latest_user for word in approval_words)

    def _execute_write_file(path: str, content: str) -> str:
        target = _resolve_path(workspace_root, path)
        _ensure_tool_path_allowed(target, workspace_root, write=True)
        if len(content) > MAX_TOOL_WRITE_CHARS:
            return f"Refusing to write more than {MAX_TOOL_WRITE_CHARS} chars in one tool call."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target.relative_to(workspace_root)}"

    def _execute_skill_manage(
        *,
        action: str,
        name: str = "",
        description: str = "",
        content: str = "",
        enabled: bool = True,
    ) -> str:
        op = action.strip().lower()
        if op in {"create", "update"}:
            if not name or not content:
                return "name and content are required for create/update."
            skill = skills.create_or_update(name=name, description=description, content=content, enabled=enabled)
            return f"{op}d skill: {skill.name}"
        if op == "enable":
            if not name:
                return "name is required for enable."
            skill = skills.set_enabled(name, True)
            return f"Enabled skill: {skill.name}"
        if op == "disable":
            if not name:
                return "name is required for disable."
            skill = skills.set_enabled(name, False)
            return f"Disabled skill: {skill.name}"
        if op == "delete":
            if not name:
                return "name is required for delete."
            skills.delete(name)
            return f"Deleted skill: {name}"
        return "Unsupported action."

    def _execute_skill_action(skill_name: str, action: str, args: dict) -> str:
        skill = skills.get(skill_name)
        if skill is None or not skill.enabled:
            return f"Skill is not available or not enabled: {skill_name}"

        normalized_skill = skill.name.strip().lower()
        normalized_action = action.strip().lower()
        if normalized_skill == "rag":
            if rag is None:
                return "RAG skill is disabled."
            if normalized_action == "search":
                query = str(args.get("query", "")).strip()
                top_k = int(args.get("top_k", 5))
                if not query:
                    return "RAG search requires args.query."
                try:
                    result = rag.search(query, top_k=top_k)
                    return result.render()
                except Exception as exc:  # noqa: BLE001
                    return f"RAG skill search failed: {exc}"
            if normalized_action == "index":
                path = str(args.get("path", "knowledge"))
                drop_old = bool(args.get("drop_old", False))
                target = _resolve_path(workspace_root, path)
                _ensure_tool_path_allowed(target, workspace_root)
                if not target.exists():
                    return f"Path does not exist: {path}"
                try:
                    return rag.index_paths([target], drop_old=drop_old)
                except Exception as exc:  # noqa: BLE001
                    return f"RAG skill index failed: {exc}"
            return "RAG skill supports actions: search, index."

        return f"Skill action is not executable: {skill_name}.{action}"

    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories under a workspace-relative path."""
        target = _resolve_path(workspace_root, path)
        _ensure_tool_path_allowed(target, workspace_root)
        if not target.exists():
            return f"Path does not exist: {path}"
        if target.is_file():
            return str(target.relative_to(workspace_root))
        rows = []
        for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if _is_blocked_path(item, workspace_root):
                continue
            kind = "DIR" if item.is_dir() else "FILE"
            rows.append(f"{kind}\t{item.relative_to(workspace_root)}")
            if len(rows) >= MAX_LIST_ITEMS:
                rows.append(f"... truncated after {MAX_LIST_ITEMS} items")
                break
        return "\n".join(rows) if rows else "(empty)"

    @tool
    def read_file(path: str, max_chars: int = 6000) -> str:
        """Read a text file from workspace. Use workspace-relative path."""
        target = _resolve_path(workspace_root, path)
        _ensure_tool_path_allowed(target, workspace_root)
        if not target.exists() or not target.is_file():
            return f"File not found: {path}"
        limit = max(1, min(int(max_chars), MAX_TOOL_READ_CHARS))
        with target.open("r", encoding="utf-8") as handle:
            return handle.read(limit)

    @tool
    def write_file(path: str, content: str, approved_request_id: str = "") -> str:
        """Request approval to write a text file. Use approve_pending_action to execute after user approval."""
        target = _resolve_path(workspace_root, path)
        _ensure_tool_path_allowed(target, workspace_root, write=True)
        if len(content) > MAX_TOOL_WRITE_CHARS:
            return f"Refusing to write more than {MAX_TOOL_WRITE_CHARS} chars in one tool call."
        if approved_request_id:
            if not _latest_user_approves(approved_request_id):
                return (
                    "Explicit user approval was not detected in the latest user message. "
                    f"Ask the user to approve {approved_request_id} before executing it."
                )
            item = human_loop.get_pending(thread_id_getter(), approved_request_id)
            if not item or item.get("action") != "write_file":
                return f"No pending write_file approval found for id={approved_request_id}."
            payload = item.get("payload", {})
            if payload.get("path") != path or payload.get("content") != content:
                return "Approved request payload does not match this write_file call."
            result = _execute_write_file(path, content)
            human_loop.mark_completed(thread_id_getter(), approved_request_id, result)
            return result
        item = human_loop.create_approval(
            thread_id=thread_id_getter(),
            action="write_file",
            payload={"path": path, "content": content},
            rationale=f"Write {len(content)} chars to {path}.",
        )
        return _approval_required(item, preview=f"{path} ({len(content)} chars)")

    @tool
    def search_in_files(pattern: str, path: str = ".") -> str:
        """Regex search in text files under workspace-relative path."""
        target = _resolve_path(workspace_root, path)
        _ensure_tool_path_allowed(target, workspace_root)
        if not target.exists():
            return f"Path does not exist: {path}"
        regex = re.compile(pattern, re.IGNORECASE)
        files = _iter_search_files(target, workspace_root)
        hits: list[str] = []
        for file_path in files:
            rel = file_path.relative_to(workspace_root)
            try:
                if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        hits.append(f"{rel}:{index}: {line.strip()}")
                    if len(hits) >= 80:
                        return "\n".join(hits)
            except UnicodeDecodeError:
                continue
        return "\n".join(hits) if hits else "No matches."

    @tool
    def add_memory(note: str) -> str:
        """Store durable memory note for current thread."""
        tid = thread_id_getter()
        memory.add_fact(tid, note)
        return "Memory note stored."

    @tool
    def query_memory() -> str:
        """Read current thread memory snapshot."""
        tid = thread_id_getter()
        return memory.build_context_block(tid) or "No memory yet."

    @tool
    def skill_catalog(enabled_only: bool = True) -> str:
        """List available skills and status."""
        items = skills.list_skills(enabled_only=enabled_only)
        if not items:
            return "No skills found."
        return "\n".join(
            f"- {item.name} | enabled={item.enabled} | {item.description}".strip()
            for item in items
        )

    @tool
    def skill_manage(
        action: str,
        name: str = "",
        description: str = "",
        content: str = "",
        enabled: bool = True,
        approved_request_id: str = "",
    ) -> str:
        """Request approval to create, update, enable, disable, or delete skills."""
        payload = {
            "action": action,
            "name": name,
            "description": description,
            "content": content,
            "enabled": enabled,
        }
        if approved_request_id:
            if not _latest_user_approves(approved_request_id):
                return (
                    "Explicit user approval was not detected in the latest user message. "
                    f"Ask the user to approve {approved_request_id} before executing it."
                )
            item = human_loop.get_pending(thread_id_getter(), approved_request_id)
            if not item or item.get("action") != "skill_manage":
                return f"No pending skill_manage approval found for id={approved_request_id}."
            if item.get("payload") != payload:
                return "Approved request payload does not match this skill_manage call."
            result = _execute_skill_manage(**payload)
            human_loop.mark_completed(thread_id_getter(), approved_request_id, result)
            return result
        item = human_loop.create_approval(
            thread_id=thread_id_getter(),
            action="skill_manage",
            payload=payload,
            rationale=f"Run skill_manage action={action!r} name={name!r}.",
        )
        return _approval_required(item, preview=f"skill_manage {action} {name}".strip())

    @tool
    def request_user_input(question: str, context: str = "", options: list[str] | None = None) -> str:
        """Ask the user for missing information before continuing."""
        item = human_loop.create_clarification(
            thread_id=thread_id_getter(),
            question=question,
            context=context,
            options=options or [],
        )
        return (
            "HUMAN_INPUT_REQUIRED\n"
            f"id={item['id']}\n"
            f"question={item['question']}\n"
            "Stop and ask the user this question."
        )

    @tool
    def list_pending_human_requests() -> str:
        """List pending human clarification and approval requests for this thread."""
        items = human_loop.list_pending(thread_id_getter())
        if not items:
            return "No pending human requests."
        rows = []
        for item in items:
            if item.get("type") == "approval":
                rows.append(f"{item['id']} approval {item.get('action')}: {item.get('rationale')}")
            else:
                rows.append(f"{item['id']} clarification: {item.get('question')}")
        return "\n".join(rows)

    @tool
    def deny_pending_request(request_id: str, reason: str = "") -> str:
        """Mark a pending human request as denied."""
        if not human_loop.get_pending(thread_id_getter(), request_id):
            return f"No pending request found for id={request_id}."
        human_loop.mark_denied(thread_id_getter(), request_id, reason)
        return f"Denied request {request_id}."

    @tool
    def record_user_input(request_id: str, answer: str) -> str:
        """Record the user's answer to a pending clarification request."""
        item = human_loop.get_pending(thread_id_getter(), request_id)
        if not item:
            return f"No pending request found for id={request_id}."
        if item.get("type") != "clarification":
            return f"Request {request_id} is not a clarification request."
        result = f"User answered: {answer.strip()}"
        human_loop.mark_completed(thread_id_getter(), request_id, result)
        return result

    @tool
    def approve_pending_action(request_id: str) -> str:
        """Execute an approved pending action after the user explicitly approves its id."""
        if not _latest_user_approves(request_id):
            return (
                "Explicit user approval was not detected in the latest user message. "
                f"Ask the user to approve {request_id} before executing it."
            )
        item = human_loop.get_pending(thread_id_getter(), request_id)
        if not item:
            return f"No pending request found for id={request_id}."
        if item.get("type") != "approval":
            return f"Request {request_id} is not an executable approval."
        action = item.get("action")
        payload = item.get("payload", {})
        if action == "write_file":
            result = _execute_write_file(path=payload.get("path", ""), content=payload.get("content", ""))
        elif action == "skill_manage":
            result = _execute_skill_manage(**payload)
        elif action == "use_skill":
            result = _execute_skill_action(
                skill_name=payload.get("skill_name", ""),
                action=payload.get("action", ""),
                args=payload.get("args", {}),
            )
        else:
            return f"Unsupported approval action: {action}"
        human_loop.mark_completed(thread_id_getter(), request_id, result)
        return result

    @tool
    def use_skill(
        skill_name: str,
        action: str,
        args: dict | None = None,
        approved_request_id: str = "",
    ) -> str:
        """Use an enabled skill action. RAG is available as use_skill(skill_name='rag', action='search'|'index', args={...})."""
        final_args = args or {}
        if skill_name.strip().lower() == "rag" and action.strip().lower() == "index":
            payload = {"skill_name": skill_name, "action": action, "args": final_args}
            if approved_request_id:
                if not _latest_user_approves(approved_request_id):
                    return (
                        "Explicit user approval was not detected in the latest user message. "
                        f"Ask the user to approve {approved_request_id} before executing it."
                    )
                item = human_loop.get_pending(thread_id_getter(), approved_request_id)
                if not item or item.get("action") != "use_skill":
                    return f"No pending use_skill approval found for id={approved_request_id}."
                if item.get("payload") != payload:
                    return "Approved request payload does not match this use_skill call."
                result = _execute_skill_action(skill_name=skill_name, action=action, args=final_args)
                human_loop.mark_completed(thread_id_getter(), approved_request_id, result)
                return result
            item = human_loop.create_approval(
                thread_id=thread_id_getter(),
                action="use_skill",
                payload=payload,
                rationale=f"Run skill action {skill_name}.{action}.",
            )
            return _approval_required(item, preview=f"use_skill {skill_name}.{action}")
        return _execute_skill_action(skill_name=skill_name, action=action, args=final_args)

    @tool
    def mcp_catalog() -> str:
        """List configured MCP servers and their loaded tool counts."""
        if mcp_catalog_getter is None:
            return "MCP registry is not configured."
        return mcp_catalog_getter()

    tools: list[BaseTool] = [
        list_files,
        read_file,
        write_file,
        search_in_files,
        add_memory,
        query_memory,
        skill_catalog,
        skill_manage,
        request_user_input,
        list_pending_human_requests,
        deny_pending_request,
        record_user_input,
        approve_pending_action,
        use_skill,
        mcp_catalog,
    ]

    if include_delegate and subagent_runner is not None:
        @tool
        def delegate_task(task: str, context: str = "") -> str:
            """Delegate focused sub-task to an isolated sub-agent."""
            return subagent_runner.run(task=task, context=context)

        tools.append(delegate_task)

    try:
        from langchain_tavily import TavilySearch
    except Exception:  # noqa: BLE001
        TavilySearch = None

    if TavilySearch is not None:
        try:
            tools.append(
                TavilySearch(
                    api_key=os.getenv('TAVILY_API_KEY'),
                    max_results=3,
                    topic="general"
                )
            )
        except Exception:
            pass

    return tools
