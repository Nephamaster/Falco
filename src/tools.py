from __future__ import annotations

import os
import re
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from memory import ConversationMemoryManager
from rag import MilvusRAG
from skills import SkillManager
from subagent import SubAgentRunner


load_dotenv()


def _resolve_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = (workspace_root / raw_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:  # noqa: B904
        raise ValueError("Path escapes workspace root.") from exc
    return candidate


def create_core_tools(
    *,
    workspace_root: Path,
    memory: ConversationMemoryManager,
    skills: SkillManager,
    rag: MilvusRAG | None,
    thread_id_getter,
    subagent_runner: SubAgentRunner | None = None,
    include_delegate: bool = True,
) -> list[BaseTool]:
    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories under a workspace-relative path."""
        target = _resolve_path(workspace_root, path)
        if not target.exists():
            return f"Path does not exist: {path}"
        if target.is_file():
            return str(target.relative_to(workspace_root))
        rows = []
        for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            kind = "DIR" if item.is_dir() else "FILE"
            rows.append(f"{kind}\t{item.relative_to(workspace_root)}")
        return "\n".join(rows) if rows else "(empty)"

    @tool
    def read_file(path: str, max_chars: int = 6000) -> str:
        """Read a text file from workspace. Use workspace-relative path."""
        target = _resolve_path(workspace_root, path)
        if not target.exists() or not target.is_file():
            return f"File not found: {path}"
        content = target.read_text(encoding="utf-8")
        return content[:max_chars]

    @tool
    def write_file(path: str, content: str) -> str:
        """Write text content to a file in workspace. Overwrites if file exists."""
        target = _resolve_path(workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target.relative_to(workspace_root)}"

    @tool
    def search_in_files(pattern: str, path: str = ".") -> str:
        """Regex search in text files under workspace-relative path."""
        target = _resolve_path(workspace_root, path)
        if not target.exists():
            return f"Path does not exist: {path}"
        regex = re.compile(pattern, re.IGNORECASE)
        files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
        hits: list[str] = []
        for file_path in files:
            rel = file_path.relative_to(workspace_root)
            try:
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
    ) -> str:
        """Manage skills: create, update, enable, disable, delete."""
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
    
    @tool
    def rag_search(query: str, top_k: int = 5) -> str:
        """Search local Milvus vector DB with query optimization and reranking."""
        if rag is None:
            return "RAG is disabled."
        try:
            result = rag.search(query, top_k=top_k)
            return result.render()
        except Exception as exc:  # noqa: BLE001
            return f"RAG search failed: {exc}"

    @tool
    def rag_index(path: str = "knowledge", drop_old: bool = False) -> str:
        """Index local documents into Milvus for RAG retrieval."""
        if rag is None:
            return "RAG is disabled."
        target = _resolve_path(workspace_root, path)
        if not target.exists():
            return f"Path does not exist: {path}"
        try:
            return rag.index_paths([target], drop_old=drop_old)
        except Exception as exc:  # noqa: BLE001
            return f"RAG index failed: {exc}"

    tools: list[BaseTool] = [
        list_files,
        read_file,
        write_file,
        search_in_files,
        add_memory,
        query_memory,
        skill_catalog,
        skill_manage,
    ]

    if rag is not None:
        tools.extend([rag_search, rag_index])

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
