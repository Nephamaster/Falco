from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_PREFIX_ROOTS = {
    "uploads": "uploads",
    "runtime": "runtime",
    "deliverables": "deliverables",
}


@dataclass(frozen=True)
class WorkspaceManager:
    primary_root: Path
    allowed_roots: tuple[Path, ...]
    default_working_directory: Path
    uploads_root: Path
    runtime_root: Path
    deliverables_root: Path
    thread_scoped_directories: bool = True

    def __post_init__(self) -> None:
        primary_root = self.primary_root.resolve()
        extra_roots = (
            self.uploads_root.resolve(),
            self.runtime_root.resolve(),
            self.deliverables_root.resolve(),
        )
        allowed_roots = tuple(dict.fromkeys(root.resolve() for root in (self.allowed_roots or (primary_root,)) + extra_roots))
        default_working_directory = self.default_working_directory.resolve()

        if not self._is_allowed_path(default_working_directory, allowed_roots):
            raise ValueError("Default working directory must stay within allowed roots.")

        object.__setattr__(self, "primary_root", primary_root)
        object.__setattr__(self, "allowed_roots", allowed_roots)
        object.__setattr__(self, "default_working_directory", default_working_directory)
        object.__setattr__(self, "uploads_root", self.uploads_root.resolve())
        object.__setattr__(self, "runtime_root", self.runtime_root.resolve())
        object.__setattr__(self, "deliverables_root", self.deliverables_root.resolve())

    def _sanitize_thread_id(self, thread_id: str) -> str:
        safe = "".join(ch for ch in str(thread_id or "default") if ch.isalnum() or ch in ("-", "_"))
        return safe or "default"

    def _is_allowed_path(self, path: Path, allowed_roots: tuple[Path, ...] | None = None) -> bool:
        candidate = path.resolve()
        roots = allowed_roots or self.allowed_roots
        for root in roots:
            try:
                candidate.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def is_allowed_path(self, path: Path) -> bool:
        return self._is_allowed_path(path)

    def thread_directory(self, thread_id: str, kind: str) -> Path:
        base = {
            "uploads": self.uploads_root,
            "runtime": self.runtime_root,
            "deliverables": self.deliverables_root,
        }[kind]
        if self.thread_scoped_directories:
            return (base / self._sanitize_thread_id(thread_id)).resolve()
        return base.resolve()

    def ensure_thread_directories(self, thread_id: str) -> dict[str, Path]:
        dirs = {
            "uploads": self.thread_directory(thread_id, "uploads"),
            "runtime": self.thread_directory(thread_id, "runtime"),
            "deliverables": self.thread_directory(thread_id, "deliverables"),
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def _parse_prefixed_path(self, raw_path: str) -> tuple[str | None, str]:
        for prefix in _PREFIX_ROOTS:
            marker = f"{prefix}:/"
            if raw_path.startswith(marker):
                return prefix, raw_path[len(marker):]
        return None, raw_path

    def resolve_working_directory(
        self,
        raw_path: str | Path | None = None,
        *,
        base_dir: Path | None = None,
        thread_id: str | None = None,
    ) -> Path:
        if raw_path in (None, ""):
            candidate = self.default_working_directory
        else:
            raw_text = str(raw_path)
            prefix, inner = self._parse_prefixed_path(raw_text)
            if prefix:
                if not thread_id:
                    raise ValueError("Thread id is required for thread-scoped working directory aliases.")
                base = self.thread_directory(thread_id, prefix)
                candidate = (base / inner).resolve()
            else:
                candidate = Path(raw_text).expanduser()
                base = (base_dir or self.default_working_directory).resolve()
                if not candidate.is_absolute():
                    candidate = (base / candidate).resolve()
                else:
                    candidate = candidate.resolve()
        if not self.is_allowed_path(candidate):
            raise ValueError(f"Working directory is outside allowed roots: {candidate}")
        return candidate

    def resolve_thread_path(self, thread_id: str, raw_path: str, *, cwd: Path | None = None) -> Path:
        self.ensure_thread_directories(thread_id)
        base_dir = (cwd or self.default_working_directory).resolve()
        if not self.is_allowed_path(base_dir):
            raise ValueError(f"Working directory is outside allowed roots: {base_dir}")

        prefix, inner = self._parse_prefixed_path(raw_path)
        if prefix:
            candidate = (self.thread_directory(thread_id, prefix) / inner).resolve()
        else:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
        if not self.is_allowed_path(candidate):
            raise ValueError("Path escapes allowed workspace roots.")
        return candidate

    def describe_path(self, path: Path, *, cwd: Path | None = None, thread_id: str | None = None) -> str:
        candidate = path.resolve()
        if thread_id:
            thread_dirs = self.ensure_thread_directories(thread_id)
            for prefix, root in thread_dirs.items():
                try:
                    rel = candidate.relative_to(root)
                    suffix = rel.as_posix()
                    return f"{prefix}:/{suffix}" if suffix and suffix != "." else f"{prefix}:/"
                except ValueError:
                    continue

        base_dir = (cwd or self.default_working_directory).resolve()
        try:
            rel = candidate.relative_to(base_dir)
            if str(rel) == ".":
                return "."
            return str(rel)
        except ValueError:
            try:
                return str(candidate.relative_to(self.primary_root))
            except ValueError:
                return str(candidate)

    def prompt_block(self, *, thread_id: str, cwd: Path | None = None) -> str:
        current_dir = (cwd or self.default_working_directory).resolve()
        thread_dirs = self.ensure_thread_directories(thread_id)
        lines = ["<working_environment>"]
        lines.append(f"- Current working directory: {current_dir}")
        lines.append(f"- Primary workspace root: {self.primary_root}")
        lines.append("- Allowed read/write roots:")
        for root in self.allowed_roots:
            lines.append(f"  - {root}")
        lines.append(f"- Thread uploads directory: {thread_dirs['uploads']} (alias `uploads:/`)")
        lines.append(f"- Thread runtime directory: {thread_dirs['runtime']} (alias `runtime:/`)")
        lines.append(f"- Thread deliverables directory: {thread_dirs['deliverables']} (alias `deliverables:/`)")
        lines.append("- User-uploaded files belong in `uploads:/`. Treat them as inputs unless explicitly asked to modify them.")
        lines.append("- Intermediate agent work may use the current working directory, but sub-agent outputs must be written under `runtime:/`.")
        lines.append("- Decide for yourself whether subagents are needed. Use them for bounded subproblems that can be solved and handed back through Markdown files.")
        lines.append("- Use the subagent workflow in this order: create subagent tasks, run the pending subagent tasks, then read or collect their `result.md` files before writing your final answer.")
        lines.append("- Subagents do not hand results back through chat text. Their handoff contract is the Markdown result file in their worker directory under `runtime:/subagents/...`.")
        lines.append("- Final files for the user must be written under `deliverables:/` using the final-output workflow.")
        lines.append("- Relative paths resolve from the current working directory.")
        lines.append("- Do not claim a file was written unless a tool confirms the write result.")
        lines.append("- Stay inside the allowed roots. If a target path is unclear, clarify before acting.")
        lines.append("</working_environment>")
        return "\n".join(lines)
