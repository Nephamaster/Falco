from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    enabled: bool
    path: Path
    content: str


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower()).strip("-")
    return cleaned or "skill"


class SkillManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_default_skill()
        self._ensure_rag_skill()

    def _ensure_default_skill(self) -> None:
        default_path = self.root / "default_planning.md"
        if default_path.exists():
            return
        default_path.write_text(
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
                ]
            ),
            encoding="utf-8",
        )

    def _ensure_rag_skill(self) -> None:
        rag_path = self.root / "rag.md"
        if rag_path.exists():
            return
        rag_path.write_text(
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
                    "",
                    "Prefer search before answering questions that depend on local documents.",
                    "Indexing changes the knowledge base and requires human approval.",
                ]
            ),
            encoding="utf-8",
        )

    def _parse_skill(self, path: Path) -> Skill:
        raw = path.read_text(encoding="utf-8")
        name = path.stem
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

        return Skill(name=name, description=description, enabled=enabled, path=path, content=content.strip())

    def _write_skill(self, path: Path, name: str, description: str, enabled: bool, content: str) -> None:
        path.write_text(
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

    def list_skills(self, enabled_only: bool = False) -> list[Skill]:
        skills = [self._parse_skill(path) for path in sorted(self.root.glob("*.md"))]
        if enabled_only:
            skills = [skill for skill in skills if skill.enabled]
        return skills

    def get_prompt_block(self, max_chars: int = 5000) -> str:
        blocks: list[str] = []
        for skill in self.list_skills(enabled_only=True):
            entry = f"[Skill: {skill.name}]\n{skill.content}".strip()
            blocks.append(entry)
        merged = "\n\n".join(blocks)
        return merged[:max_chars]

    def create_or_update(self, name: str, description: str, content: str, enabled: bool = True) -> Skill:
        skill_name = _slugify(name)
        path = self.root / f"{skill_name}.md"
        self._write_skill(path, skill_name, description, enabled, content)
        return self._parse_skill(path)

    def set_enabled(self, name: str, enabled: bool) -> Skill:
        target = self.get(name)
        if not target:
            raise ValueError(f"Skill not found: {name}")
        self._write_skill(target.path, target.name, target.description, enabled, target.content)
        return self._parse_skill(target.path)

    def delete(self, name: str) -> None:
        target = self.get(name)
        if not target:
            raise ValueError(f"Skill not found: {name}")
        target.path.unlink()

    def get(self, name: str) -> Skill | None:
        normalized = _slugify(name)
        for skill in self.list_skills(enabled_only=False):
            if _slugify(skill.name) == normalized:
                return skill
        return None
