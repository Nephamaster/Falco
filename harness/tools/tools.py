from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from harness.agents.human_loop import HumanLoopManager
from harness.agents.memory.manager import ConversationMemoryManager
from harness.agents.subagent_tasks import SubAgentTaskManager
from harness.rag import MilvusRAG
from harness.skills.skills import SkillManager
from harness.tokenization import count_tokens, truncate_tokens
from harness.workspace import WorkspaceManager

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


load_dotenv()

MAX_TOOL_READ_TOKENS = 100_000
MAX_TOOL_WRITE_TOKENS = 200_000
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


def _resolve_path(
    workspace: WorkspaceManager,
    thread_id: str,
    raw_path: str,
    *,
    cwd: Path | None = None,
) -> Path:
    return workspace.resolve_thread_path(thread_id, raw_path, cwd=cwd)


def _is_blocked_path(path: Path, workspace: WorkspaceManager) -> bool:
    if not workspace.is_allowed_path(path):
        return True

    candidate = path.resolve()
    rel = None
    for root in workspace.allowed_roots:
        try:
            rel = candidate.relative_to(root)
            break
        except ValueError:
            continue
    if rel is None:
        return True

    parts = set(rel.parts)
    if parts & BLOCKED_PATH_PARTS:
        return True
    if path.name in BLOCKED_FILE_NAMES:
        return True
    return path.suffix.lower() in BLOCKED_FILE_SUFFIXES


def _ensure_tool_path_allowed(path: Path, workspace: WorkspaceManager, *, write: bool = False) -> None:
    if _is_blocked_path(path, workspace):
        action = "write" if write else "access"
        rel = workspace.describe_path(path)
        raise ValueError(f"Refusing to {action} sensitive or generated path: {rel}")


def _iter_search_files(root: Path, workspace: WorkspaceManager) -> list[Path]:
    if root.is_file():
        return [root] if not _is_blocked_path(root, workspace) else []

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_blocked_path(current / name, workspace)
        ]
        for name in filenames:
            file_path = current / name
            if _is_blocked_path(file_path, workspace):
                continue
            files.append(file_path)
            if len(files) >= MAX_SEARCH_FILES:
                return files
    return files


def _execute_write_file_impl(
    workspace: WorkspaceManager,
    path: str,
    content: str,
    *,
    cwd: Path | None = None,
    thread_id: str,
) -> str:
    target = _resolve_path(workspace, thread_id, path, cwd=cwd)
    _ensure_tool_path_allowed(target, workspace, write=True)
    normalized_content = _normalize_text_content(content)
    content_tokens = count_tokens(normalized_content)
    if content_tokens > MAX_TOOL_WRITE_TOKENS:
        return f"Refusing to write more than {MAX_TOOL_WRITE_TOKENS} tokens in one tool call."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(normalized_content, encoding="utf-8")
    return f"Wrote {content_tokens} tokens to {target}"


def _normalize_text_content(content: str) -> str:
    text = str(content or "")
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    if "\t" not in text and "\\t" in text:
        text = text.replace("\\t", "\t")
    return text


def _execute_skill_manage_impl(
    skills: SkillManager,
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


def _execute_skill_action_impl(
    skills: SkillManager,
    rag: MilvusRAG | None,
    workspace: WorkspaceManager,
    *,
    thread_id: str,
    cwd: Path | None = None,
    skill_name: str,
    action: str,
    args: dict,
) -> str:
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
            target = _resolve_path(workspace, thread_id, path, cwd=cwd)
            _ensure_tool_path_allowed(target, workspace)
            if not target.exists():
                return f"Path does not exist: {path}"
            try:
                return rag.index_paths([target], drop_old=drop_old)
            except Exception as exc:  # noqa: BLE001
                return f"RAG skill index failed: {exc}"
        return "RAG skill supports actions: search, index."

    return f"Skill action is not executable: {skill_name}.{action}"


def execute_pending_approval(
    *,
    thread_id: str,
    request_id: str,
    human_loop: HumanLoopManager,
    workspace: WorkspaceManager,
    current_working_directory: Path | None,
    skills: SkillManager,
    rag: MilvusRAG | None,
) -> str:
    item = human_loop.get_pending(thread_id, request_id)
    if not item:
        return f"No pending request found for id={request_id}."
    if item.get("type") != "approval":
        return f"Request {request_id} is not an executable approval."

    action = item.get("action")
    payload = item.get("payload", {})
    if action == "write_file":
        result = _execute_write_file_impl(
            workspace,
            path=payload.get("path", ""),
            content=payload.get("content", ""),
            cwd=current_working_directory,
            thread_id=thread_id,
        )
    elif action == "write_final_file":
        result = _execute_write_file_impl(
            workspace,
            path=payload.get("path", ""),
            content=payload.get("content", ""),
            cwd=current_working_directory,
            thread_id=thread_id,
        )
    elif action == "skill_manage":
        result = _execute_skill_manage_impl(skills, **payload)
    elif action == "use_skill":
        result = _execute_skill_action_impl(
            skills,
            rag,
            workspace,
            thread_id=thread_id,
            cwd=current_working_directory,
            skill_name=payload.get("skill_name", ""),
            action=payload.get("action", ""),
            args=payload.get("args", {}),
        )
    else:
        return f"Unsupported approval action: {action}"

    human_loop.mark_completed(thread_id, request_id, result)
    return result


def create_core_tools(
    *,
    workspace: WorkspaceManager,
    memory: ConversationMemoryManager,
    human_loop: HumanLoopManager,
    skills: SkillManager,
    rag: MilvusRAG | None,
    thread_id_getter,
    latest_user_getter,
    working_directory_getter,
    working_directory_setter=None,
    working_directory_persistor=None,
    resume_input_getter=None,
    mcp_catalog_getter=None,
    subagent_task_manager: SubAgentTaskManager | None = None,
    subagent_task_runner=None,
    max_subagents: int = 4,
    include_delegate: bool = True,
) -> list[BaseTool]:
    def _current_thread_id() -> str:
        return str(thread_id_getter() or "default")

    def _current_working_directory() -> Path:
        return workspace.resolve_working_directory(
            working_directory_getter(),
            thread_id=_current_thread_id(),
        )

    def _thread_dirs() -> dict[str, Path]:
        return workspace.ensure_thread_directories(_current_thread_id())

    def _deliverable_alias(path: str) -> str:
        if path.startswith("deliverables:/"):
            return path
        return f"deliverables:/{path.lstrip('/')}"

    def _build_approval_question(action: str, rationale: str) -> str:
        normalized_action = action.strip() or "this action"
        normalized_rationale = rationale.strip()
        if normalized_rationale:
            return f"I am ready to execute {normalized_action}. Reason: {normalized_rationale} Approve?"
        return f"I am ready to execute {normalized_action}. Approve?"

    def _approval_required(item: dict, preview: str = "") -> str:
        lines = [
            "HUMAN_APPROVAL_REQUIRED",
            f"id={item['id']}",
            f"action={item.get('action', '')}",
            f"question={item.get('question', '')}",
            f"rationale={item.get('rationale', '')}",
        ]
        if preview:
            lines.append(f"preview={preview}")
        lines.append("Ask the user to approve or deny this id before continuing.")
        return "\n".join(lines)

    def _latest_user_approves(request_id: str) -> bool:
        latest_user = latest_user_getter().lower()
        resume_input = str(resume_input_getter() or "").lower() if resume_input_getter is not None else ""
        approval_words = ("approve", "approved", "yes", "confirm")
        if request_id.lower() in latest_user and any(word in latest_user for word in approval_words):
            return True
        return bool(resume_input and any(word in resume_input for word in approval_words))

    def _execute_write_file(path: str, content: str) -> str:
        return _execute_write_file_impl(workspace, path, content, cwd=_current_working_directory(), thread_id=_current_thread_id())

    def _execute_skill_manage(
        *,
        action: str,
        name: str = "",
        description: str = "",
        content: str = "",
        enabled: bool = True,
    ) -> str:
        return _execute_skill_manage_impl(
            skills,
            action=action,
            name=name,
            description=description,
            content=content,
            enabled=enabled,
        )

    def _execute_skill_action(skill_name: str, action: str, args: dict) -> str:
        return _execute_skill_action_impl(
            skills,
            rag,
            workspace,
            thread_id=_current_thread_id(),
            cwd=_current_working_directory(),
            skill_name=skill_name,
            action=action,
            args=args,
        )

    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories under a workspace-relative path."""
        target = _resolve_path(workspace, _current_thread_id(), path, cwd=_current_working_directory())
        _ensure_tool_path_allowed(target, workspace)
        if not target.exists():
            return f"Path does not exist: {path}"
        if target.is_file():
            return str(target)
        rows = []
        for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if _is_blocked_path(item, workspace):
                continue
            kind = "DIR" if item.is_dir() else "FILE"
            rows.append(f"{kind}\t{item}")
            if len(rows) >= MAX_LIST_ITEMS:
                rows.append(f"... truncated after {MAX_LIST_ITEMS} items")
                break
        return "\n".join(rows) if rows else "(empty)"

    @tool
    def read_file(path: str, max_tokens: int = 6000) -> str:
        """Read a text file from workspace. Use workspace-relative path."""
        target = _resolve_path(workspace, _current_thread_id(), path, cwd=_current_working_directory())
        _ensure_tool_path_allowed(target, workspace)
        if not target.exists() or not target.is_file():
            return f"File not found: {path}"
        limit = max(1, min(int(max_tokens), MAX_TOOL_READ_TOKENS))
        with target.open("r", encoding="utf-8") as handle:
            content = handle.read()
        return truncate_tokens(content, limit)

    @tool
    def write_file(path: str, content: str, approved_request_id: str = "") -> str:
        """Request approval to write a text file. Use approve_pending_action to execute after user approval."""
        target = _resolve_path(workspace, _current_thread_id(), path, cwd=_current_working_directory())
        _ensure_tool_path_allowed(target, workspace, write=True)
        content_tokens = count_tokens(content)
        if content_tokens > MAX_TOOL_WRITE_TOKENS:
            return f"Refusing to write more than {MAX_TOOL_WRITE_TOKENS} tokens in one tool call."
        if approved_request_id:
            if not _latest_user_approves(approved_request_id):
                return (
                    "Explicit user approval was not detected in the latest user message. "
                    f"Ask the user to approve {approved_request_id} before executing it."
                )
            item = human_loop.get_pending(_current_thread_id(), approved_request_id)
            if not item or item.get("action") != "write_file":
                return f"No pending write_file approval found for id={approved_request_id}."
            payload = item.get("payload", {})
            if payload.get("path") != path or payload.get("content") != content:
                return "Approved request payload does not match this write_file call."
            result = _execute_write_file(path, content)
            human_loop.mark_completed(_current_thread_id(), approved_request_id, result)
            return result
        item = human_loop.create_approval(
            thread_id=_current_thread_id(),
            action="write_file",
            payload={"path": path, "content": content},
            question=_build_approval_question("write_file", f"Write {content_tokens} tokens to {target}."),
            rationale=f"Write {content_tokens} tokens to {target}.",
        )
        return _approval_required(item, preview=f"{target} ({content_tokens} tokens)")

    @tool
    def write_final_file(path: str, content: str, approved_request_id: str = "") -> str:
        """Request approval to write a final user-facing file into the thread deliverables directory."""
        deliverable_path = _deliverable_alias(path)
        target = _resolve_path(workspace, _current_thread_id(), deliverable_path, cwd=_current_working_directory())
        _ensure_tool_path_allowed(target, workspace, write=True)
        content_tokens = count_tokens(content)
        if content_tokens > MAX_TOOL_WRITE_TOKENS:
            return f"Refusing to write more than {MAX_TOOL_WRITE_TOKENS} tokens in one tool call."
        if approved_request_id:
            if not _latest_user_approves(approved_request_id):
                return (
                    "Explicit user approval was not detected in the latest user message. "
                    f"Ask the user to approve {approved_request_id} before executing it."
                )
            item = human_loop.get_pending(_current_thread_id(), approved_request_id)
            if not item or item.get("action") != "write_final_file":
                return f"No pending write_final_file approval found for id={approved_request_id}."
            payload = item.get("payload", {})
            if payload.get("path") != deliverable_path or payload.get("content") != content:
                return "Approved request payload does not match this write_final_file call."
            result = _execute_write_file_impl(
                workspace,
                deliverable_path,
                content,
                cwd=_current_working_directory(),
                thread_id=_current_thread_id(),
            )
            human_loop.mark_completed(_current_thread_id(), approved_request_id, result)
            return result
        item = human_loop.create_approval(
            thread_id=_current_thread_id(),
            action="write_final_file",
            payload={"path": deliverable_path, "content": content},
            question=_build_approval_question("write_final_file", f"Write {content_tokens} tokens to {target}."),
            rationale=f"Write {content_tokens} tokens to {target}.",
        )
        return _approval_required(item, preview=f"{target} ({content_tokens} tokens)")

    @tool
    def search_in_files(pattern: str, path: str = ".") -> str:
        """Regex search in text files under workspace-relative path."""
        target = _resolve_path(workspace, _current_thread_id(), path, cwd=_current_working_directory())
        _ensure_tool_path_allowed(target, workspace)
        if not target.exists():
            return f"Path does not exist: {path}"
        regex = re.compile(pattern, re.IGNORECASE)
        files = _iter_search_files(target, workspace)
        hits: list[str] = []
        for file_path in files:
            try:
                if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        hits.append(f"{file_path}:{index}: {line.strip()}")
                    if len(hits) >= 80:
                        return "\n".join(hits)
            except UnicodeDecodeError:
                continue
        return "\n".join(hits) if hits else "No matches."

    @tool
    def show_working_directory() -> str:
        """Show the current working directory and allowed workspace roots."""
        cwd = _current_working_directory()
        thread_dirs = _thread_dirs()
        allowed = "\n".join(f"- {root}" for root in workspace.allowed_roots)
        return (
            f"Current working directory: {cwd}\n"
            f"Thread uploads directory: {thread_dirs['uploads']}\n"
            f"Thread runtime directory: {thread_dirs['runtime']}\n"
            f"Thread deliverables directory: {thread_dirs['deliverables']}\n"
            f"Primary workspace root: {workspace.primary_root}\n"
            f"Allowed workspace roots:\n{allowed}"
        )

    @tool
    def change_working_directory(path: str) -> str:
        """Change the current working directory to an allowed path."""
        if working_directory_setter is None:
            return "Changing the working directory is not supported in this context."
        target = workspace.resolve_working_directory(
            path,
            base_dir=_current_working_directory(),
            thread_id=_current_thread_id(),
        )
        working_directory_setter(str(target))
        if working_directory_persistor is not None:
            working_directory_persistor(_current_thread_id(), str(target))
        return f"Current working directory changed to {target}"

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
            item = human_loop.get_pending(_current_thread_id(), approved_request_id)
            if not item or item.get("action") != "skill_manage":
                return f"No pending skill_manage approval found for id={approved_request_id}."
            if item.get("payload") != payload:
                return "Approved request payload does not match this skill_manage call."
            result = _execute_skill_manage(**payload)
            human_loop.mark_completed(_current_thread_id(), approved_request_id, result)
            return result
        item = human_loop.create_approval(
            thread_id=_current_thread_id(),
            action="skill_manage",
            payload=payload,
            question=_build_approval_question(
                "skill_manage",
                f"Run skill_manage action={action!r} name={name!r}.",
            ),
            rationale=f"Run skill_manage action={action!r} name={name!r}.",
        )
        return _approval_required(item, preview=f"skill_manage {action} {name}".strip())

    @tool
    def request_user_input(question: str, context: str = "", options: list[str] | None = None) -> str:
        """Ask the user for missing information before continuing."""
        item = human_loop.create_clarification(
            thread_id=_current_thread_id(),
            question=question,
            clarification_type="missing_info",
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
    def ask_clarification(
        question: str,
        clarification_type: str = "missing_info",
        context: str = "",
        options: list[str] | None = None,
    ) -> str:
        """Ask the user a clarification question before proceeding. Use clarification_type such as missing_info, ambiguous_requirement, approach_choice, risk_confirmation, or suggestion."""
        item = human_loop.create_clarification(
            thread_id=_current_thread_id(),
            question=question,
            clarification_type=clarification_type,
            context=context,
            options=options or [],
        )
        clarification_kind = clarification_type.strip() or "missing_info"
        return (
            "HUMAN_INPUT_REQUIRED\n"
            f"id={item['id']}\n"
            f"clarification_type={clarification_kind}\n"
            f"question={item['question']}\n"
            "Stop and ask the user this question."
        )

    @tool
    def list_pending_human_requests() -> str:
        """List pending human clarification and approval requests for this thread."""
        items = human_loop.list_pending(_current_thread_id())
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
        if not human_loop.get_pending(_current_thread_id(), request_id):
            return f"No pending request found for id={request_id}."
        human_loop.mark_denied(_current_thread_id(), request_id, reason)
        return f"Denied request {request_id}."

    @tool
    def record_user_input(request_id: str, answer: str) -> str:
        """Record the user's answer to a pending clarification request."""
        item = human_loop.get_pending(_current_thread_id(), request_id)
        if not item:
            return f"No pending request found for id={request_id}."
        if item.get("type") != "clarification":
            return f"Request {request_id} is not a clarification request."
        result = f"User answered: {answer.strip()}"
        human_loop.mark_completed(_current_thread_id(), request_id, result)
        return result

    @tool
    def approve_pending_action(request_id: str) -> str:
        """Execute an approved pending action after the user explicitly approves its id."""
        if not _latest_user_approves(request_id):
            return (
                "Explicit user approval was not detected in the latest user message. "
                f"Ask the user to approve {request_id} before executing it."
            )
        return execute_pending_approval(
            thread_id=_current_thread_id(),
            request_id=request_id,
            human_loop=human_loop,
            workspace=workspace,
            current_working_directory=_current_working_directory(),
            skills=skills,
            rag=rag,
        )

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
                item = human_loop.get_pending(_current_thread_id(), approved_request_id)
                if not item or item.get("action") != "use_skill":
                    return f"No pending use_skill approval found for id={approved_request_id}."
                if item.get("payload") != payload:
                    return "Approved request payload does not match this use_skill call."
                result = _execute_skill_action(skill_name=skill_name, action=action, args=final_args)
                human_loop.mark_completed(_current_thread_id(), approved_request_id, result)
                return result
            item = human_loop.create_approval(
                thread_id=_current_thread_id(),
                action="use_skill",
                payload=payload,
                question=_build_approval_question("use_skill", f"Run skill action {skill_name}.{action}."),
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
        write_final_file,
        search_in_files,
        show_working_directory,
        change_working_directory,
        add_memory,
        query_memory,
        skill_catalog,
        skill_manage,
        ask_clarification,
        request_user_input,
        list_pending_human_requests,
        deny_pending_request,
        record_user_input,
        approve_pending_action,
        use_skill,
        mcp_catalog,
    ]

    if include_delegate and subagent_task_manager is not None and subagent_task_runner is not None:
        @tool
        def delegate_task(task: str, context: str = "", expected_output: str = "") -> str:
            """Create a subagent task handle. The worker will later write its result into a runtime Markdown file."""
            runtime_dir = _thread_dirs()["runtime"]
            active_tasks = subagent_task_manager.count_active_tasks(runtime_dir=runtime_dir)
            if active_tasks >= max_subagents:
                return f"Refusing to create more than {max_subagents} active subagent tasks for this thread."
            record = subagent_task_manager.create_task(
                thread_id=_current_thread_id(),
                runtime_dir=runtime_dir,
                task=task,
                context=context,
                expected_output=expected_output,
            )
            task_alias = workspace.describe_path(Path(record["task_path"]), thread_id=_current_thread_id())
            result_alias = workspace.describe_path(Path(record["result_path"]), thread_id=_current_thread_id())
            return (
                f"Created subagent task {record['worker_id']}.\n"
                f"Task file: {task_alias}\n"
                f"Result file: {result_alias}\n"
                "You can create more tasks first, then call run_subagent_tasks, and finally read results from the result files."
            )

        @tool
        def list_subagent_tasks(status: str = "") -> str:
            """List subagent tasks for this thread."""
            runtime_dir = _thread_dirs()["runtime"]
            items = subagent_task_manager.list_tasks(runtime_dir=runtime_dir, status=status.strip())
            if not items:
                return "No subagent tasks found."
            rows = []
            for item in items:
                result_alias = workspace.describe_path(Path(item["result_path"]), thread_id=_current_thread_id())
                rows.append(
                    f"{item['worker_id']} | status={item.get('status', 'unknown')} | result={result_alias} | task={item.get('task', '')}"
                )
            return "\n".join(rows)

        @tool
        def run_subagent_tasks(worker_ids: list[str] | None = None) -> str:
            """Run pending subagent tasks synchronously. Use this after creating one or more delegated tasks."""
            reports = subagent_task_runner(_current_thread_id(), worker_ids or [])
            if not reports:
                return "No matching pending subagent tasks were executed."
            rows = []
            for report in reports:
                line = f"{report['worker_id']} | status={report['status']} | result={report['result_path']}"
                if report.get("message"):
                    line += f" | message={report['message']}"
                rows.append(line)
            return "\n".join(rows)

        @tool
        def read_subagent_result(worker_id: str) -> str:
            """Read one completed subagent result Markdown file."""
            runtime_dir = _thread_dirs()["runtime"]
            try:
                return subagent_task_manager.read_result(runtime_dir=runtime_dir, worker_id=worker_id.strip())
            except FileNotFoundError as exc:
                return str(exc)

        @tool
        def collect_subagent_results(worker_ids: list[str] | None = None) -> str:
            """Collect multiple completed subagent Markdown results into a single text block for synthesis."""
            runtime_dir = _thread_dirs()["runtime"]
            records = subagent_task_manager.list_tasks(runtime_dir=runtime_dir)
            if worker_ids:
                wanted = {item.strip() for item in worker_ids if item.strip()}
                records = [item for item in records if item.get("worker_id") in wanted]
            records = [item for item in records if item.get("status") == "completed"]
            if not records:
                return "No completed subagent results found."
            chunks: list[str] = []
            for item in records:
                worker_id = str(item["worker_id"])
                try:
                    content = subagent_task_manager.read_result(runtime_dir=runtime_dir, worker_id=worker_id).strip()
                except FileNotFoundError:
                    continue
                if not content:
                    continue
                chunks.append(f"## {worker_id}\n{content}")
            return "\n\n".join(chunks) if chunks else "No completed subagent results found."

        tools.extend(
            [
                delegate_task,
                list_subagent_tasks,
                run_subagent_tasks,
                read_subagent_result,
                collect_subagent_results,
            ]
        )

    try:
        from langchain_tavily import TavilySearch
    except Exception:  # noqa: BLE001
        TavilySearch = None

    if TavilySearch is not None:
        try:
            tools.append(
                TavilySearch(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    max_results=3,
                    topic="general",
                )
            )
        except Exception:
            pass

    return tools


def create_subagent_tools(
    *,
    workspace: WorkspaceManager,
    thread_id: str,
    worker_root: Path,
) -> list[BaseTool]:
    def _resolve_worker_path(path: str) -> Path:
        return workspace.resolve_thread_path(thread_id, path, cwd=worker_root)

    def _ensure_within_worker_root(path: Path) -> None:
        candidate = path.resolve()
        try:
            candidate.relative_to(worker_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Subagent writes must stay inside the worker directory: {worker_root}") from exc

    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories visible to the worker. Relative paths use the worker directory."""
        target = _resolve_worker_path(path)
        _ensure_tool_path_allowed(target, workspace)
        if not target.exists():
            return f"Path does not exist: {path}"
        if target.is_file():
            return str(target)
        rows = []
        for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if _is_blocked_path(item, workspace):
                continue
            kind = "DIR" if item.is_dir() else "FILE"
            rows.append(f"{kind}\t{item}")
            if len(rows) >= MAX_LIST_ITEMS:
                rows.append(f"... truncated after {MAX_LIST_ITEMS} items")
                break
        return "\n".join(rows) if rows else "(empty)"

    @tool
    def read_file(path: str, max_tokens: int = 6000) -> str:
        """Read a text file accessible to the worker."""
        target = _resolve_worker_path(path)
        _ensure_tool_path_allowed(target, workspace)
        if not target.exists() or not target.is_file():
            return f"File not found: {path}"
        limit = max(1, min(int(max_tokens), MAX_TOOL_READ_TOKENS))
        return truncate_tokens(target.read_text(encoding="utf-8"), limit)

    @tool
    def search_in_files(pattern: str, path: str = ".") -> str:
        """Regex search in files accessible to the worker."""
        target = _resolve_worker_path(path)
        _ensure_tool_path_allowed(target, workspace)
        if not target.exists():
            return f"Path does not exist: {path}"
        regex = re.compile(pattern, re.IGNORECASE)
        files = _iter_search_files(target, workspace)
        hits: list[str] = []
        for file_path in files:
            try:
                if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        hits.append(f"{file_path}:{index}: {line.strip()}")
                    if len(hits) >= 80:
                        return "\n".join(hits)
            except UnicodeDecodeError:
                continue
        return "\n".join(hits) if hits else "No matches."

    @tool
    def show_working_directory() -> str:
        """Show the worker directory and allowed roots."""
        allowed = "\n".join(f"- {root}" for root in workspace.allowed_roots)
        return (
            f"Current working directory: {worker_root}\n"
            f"Worker directory: {worker_root}\n"
            f"Allowed workspace roots:\n{allowed}"
        )

    @tool
    def write_runtime_file(path: str, content: str) -> str:
        """Write a runtime Markdown or text file inside the worker directory. Use this for the final worker report and any artifacts."""
        target = _resolve_worker_path(path)
        _ensure_within_worker_root(target)
        if target.name in {"task.md", "status.json"}:
            return f"Refusing to overwrite reserved worker file: {target.name}"
        _ensure_tool_path_allowed(target, workspace, write=True)
        normalized_content = _normalize_text_content(content)
        content_tokens = count_tokens(normalized_content)
        if content_tokens > MAX_TOOL_WRITE_TOKENS:
            return f"Refusing to write more than {MAX_TOOL_WRITE_TOKENS} tokens in one tool call."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(normalized_content, encoding="utf-8")
        return f"Wrote {content_tokens} tokens to {target}"

    return [
        list_files,
        read_file,
        search_in_files,
        show_working_directory,
        write_runtime_file,
    ]
