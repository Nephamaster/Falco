from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from harness.tokenization import truncate_tokens


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    enabled: bool
    root: Path
    prompt_path: Path
    scripts_dir: Path | None
    content: str
    scope: str


@dataclass(frozen=True)
class SkillExecutionContext:
    settings: Any
    llm: Any
    workspace: Any
    thread_id: str
    working_directory: Path | None


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower()).strip("-")
    return cleaned or "skill"


class SkillManager:
    def __init__(self, public_root: Path, user_roots: list[Path] | tuple[Path, ...]) -> None:
        self.public_root = public_root
        self.user_roots = tuple(user_roots)
        self.public_root.mkdir(parents=True, exist_ok=True)
        for root in self.user_roots:
            root.mkdir(parents=True, exist_ok=True)
        self._ensure_builtin_public_skills()

    def _ensure_builtin_public_skills(self) -> None:
        default_dir = self.public_root / "default_planning"
        default_dir.mkdir(parents=True, exist_ok=True)
        default_prompt = default_dir / "SKILL.md"
        if not default_prompt.exists():
            default_prompt.write_text(
                "\n".join(
                    [
                        "---",
                        "name: default_planning",
                        "description: A concise workflow for decomposition and validation.",
                        "enabled: true",
                        "---",
                        "1) Clarify the user objective and hard constraints.",
                        "2) Build the smallest viable execution plan.",
                        "3) Execute with tool-first actions.",
                        "4) Validate outputs before returning.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

        rag_dir = self.public_root / "rag"
        rag_dir.mkdir(parents=True, exist_ok=True)
        rag_prompt = rag_dir / "SKILL.md"
        if not rag_prompt.exists():
            rag_prompt.write_text(
                "\n".join(
                    [
                        "---",
                        "name: rag",
                        "description: Local knowledge retrieval and indexing skill.",
                        "enabled: true",
                        "---",
                        "Use this skill when the task needs local knowledge-base evidence or indexing.",
                        "",
                        "Actions available through `use_skill`:",
                        "- search: args={\"query\": \"...\", \"top_k\": 5}",
                        "- index: args={\"path\": \"knowledge\", \"drop_old\": false}",
                        "- refresh_source: reserved for future incremental refresh support",
                        "- remove_source: reserved for future source-removal support",
                        "- status: reserved for future index-status inspection",
                        "",
                        "Prefer search before answering questions that depend on local documents.",
                        "Indexing changes the knowledge base and requires human approval.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    def _parse_skill_dir(self, root: Path, *, scope: str) -> Skill | None:
        prompt_path = root / "SKILL.md"
        if not prompt_path.exists():
            return None
        raw = prompt_path.read_text(encoding="utf-8")
        name = root.name
        description = ""
        enabled = True
        content = raw

        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) == 3:
                _, head, body = parts
                content = body.strip()
                for line in head.splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "name":
                        name = value or name
                    elif key == "description":
                        description = value
                    elif key == "enabled":
                        enabled = value.lower() == "true"

        scripts_dir = root / "scripts"
        return Skill(
            name=name,
            description=description,
            enabled=enabled,
            root=root,
            prompt_path=prompt_path,
            scripts_dir=scripts_dir if scripts_dir.exists() else None,
            content=content.strip(),
            scope=scope,
        )

    def _write_skill_dir(self, root: Path, name: str, description: str, enabled: bool, content: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    f"name: {name}",
                    f"description: {description}",
                    f"enabled: {'true' if enabled else 'false'}",
                    "---",
                    content.strip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "scripts").mkdir(parents=True, exist_ok=True)

    def _all_skill_roots(self) -> list[tuple[Path, str]]:
        roots: list[tuple[Path, str]] = [(self.public_root, "public")]
        roots.extend((root, "user") for root in self.user_roots)
        return roots

    def list_skills(self, enabled_only: bool = False) -> list[Skill]:
        seen: dict[str, Skill] = {}
        for root, scope in self._all_skill_roots():
            if not root.exists():
                continue
            for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
                if not child.is_dir():
                    continue
                skill = self._parse_skill_dir(child, scope=scope)
                if skill is None:
                    continue
                seen[_slugify(skill.name)] = skill
        skills = list(seen.values())
        if enabled_only:
            skills = [skill for skill in skills if skill.enabled]
        return skills

    def get_prompt_block(self, max_tokens: int = 5000, exclude_names: set[str] | None = None) -> str:
        hidden = {_slugify(name) for name in (exclude_names or set())}
        blocks: list[str] = []
        for skill in self.list_skills(enabled_only=True):
            if _slugify(skill.name) in hidden:
                continue
            entry = f"[Skill: {skill.name}]\n{skill.content}".strip()
            blocks.append(entry)
        merged = "\n\n".join(blocks)
        return truncate_tokens(merged, max_tokens)

    def create_or_update(self, name: str, description: str, content: str, enabled: bool = True) -> Skill:
        skill_name = _slugify(name)
        target_root = self.user_roots[0] if self.user_roots else self.public_root
        skill_root = target_root / skill_name
        self._write_skill_dir(skill_root, skill_name, description, enabled, content)
        parsed = self._parse_skill_dir(skill_root, scope="user")
        if parsed is None:
            raise ValueError(f"Failed to create skill: {name}")
        return parsed

    def set_enabled(self, name: str, enabled: bool) -> Skill:
        target = self.get(name)
        if not target:
            raise ValueError(f"Skill not found: {name}")
        self._write_skill_dir(target.root, target.name, target.description, enabled, target.content)
        parsed = self._parse_skill_dir(target.root, scope=target.scope)
        if parsed is None:
            raise ValueError(f"Failed to update skill: {name}")
        return parsed

    def delete(self, name: str) -> None:
        target = self.get(name)
        if not target:
            raise ValueError(f"Skill not found: {name}")
        if target.scope == "public":
            raise ValueError(f"Cannot delete public skill: {name}")
        for path in sorted(target.root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        target.root.rmdir()

    def get(self, name: str) -> Skill | None:
        normalized = _slugify(name)
        for skill in self.list_skills(enabled_only=False):
            if _slugify(skill.name) == normalized:
                return skill
        return None

    def execute(self, skill_name: str, action: str, args: dict, context: SkillExecutionContext) -> str:
        skill = self.get(skill_name)
        if skill is None or not skill.enabled:
            return f"Skill is not available or not enabled: {skill_name}"
        runtime_path = (skill.scripts_dir / "runtime.py") if skill.scripts_dir else None
        if runtime_path is None or not runtime_path.exists():
            return f"Skill action is not executable: {skill.name}.{action}"

        spec = spec_from_file_location(f"falco_skill_{_slugify(skill.name)}_{_slugify(action)}", runtime_path)
        if spec is None or spec.loader is None:
            return f"Skill runtime could not be loaded: {skill.name}"
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        execute = getattr(module, "execute", None)
        if not callable(execute):
            return f"Skill runtime is missing execute(): {skill.name}"
        return str(execute(action=action, args=args, context=context))
