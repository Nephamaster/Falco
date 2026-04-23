from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


def _resolve_path(base: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_path_string(base: Path, raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    if "://" in text:
        return text
    return str(_resolve_path(base, text))


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_str(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_str_list(value: Any, *, default: list[str] | None = None) -> tuple[str, ...]:
    if value is None:
        value = default or []
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config.yaml. Please install `pyyaml`.")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return raw


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_refs(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    if isinstance(value, str) and value.startswith("$") and len(value) > 1:
        return os.getenv(value[1:], "")
    return value


@dataclass(frozen=True)
class FalcoSettings:
    config_version: int
    model: str
    api_key: str
    base_url: str | None
    workspace_root: Path
    allowed_workspace_roots: tuple[Path, ...]
    default_working_directory: Path
    uploads_root: Path
    runtime_root: Path
    deliverables_root: Path
    thread_scoped_directories: bool
    memory_root: Path
    skills_public_root: Path
    skills_user_roots: tuple[Path, ...]
    soul_path: Path | None = None
    max_context_messages: int = 12
    max_tool_steps: int = 6
    max_subagents: int = 4
    subagent_max_steps: int = 4
    tool_read_max_tokens: int = 100_000
    tool_write_max_tokens: int = 200_000
    tool_search_file_max_bytes: int = 1_000_000
    tool_search_max_files: int = 800
    tool_list_max_items: int = 300
    blocked_path_parts: tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        ".falco",
        "__pycache__",
        "node_modules",
        ".next",
        ".venv",
        "venv",
    )
    blocked_file_names: tuple[str, ...] = (
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "id_rsa",
        "id_ed25519",
    )
    blocked_file_suffixes: tuple[str, ...] = (
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    )
    memory_recent_rounds: int = 6
    memory_key_rounds: int = 4
    memory_importance_threshold: int = 7
    memory_max_rounds: int = 160
    memory_context_soft_limit_tokens: int = 7000
    memory_context_max_tokens: int = 9000
    memory_silent_turn_cooldown_rounds: int = 4
    memory_daily_half_life_days: int = 30
    memory_daily_lookback_days: int = 180
    memory_daily_retrieval_items: int = 8
    memory_evergreen_retrieval_items: int = 5
    rag_enabled: bool = True
    rag_milvus_uri: str = "./.falco/milvus/falco_rag.db"
    rag_milvus_token: str | None = None
    rag_collection: str = "falco_knowledge"
    rag_embedding_model: str = "text-embedding-3-small"
    rag_base_url: str = ""
    rag_api_key: str = ""
    rag_retrieval_mode: str = "dense"
    rag_hybrid_dense_weight: float = 0.7
    rag_hybrid_sparse_weight: float = 0.3
    rag_top_k: int = 5
    rag_fetch_k: int = 18
    rag_rerank_enabled: bool = True
    rag_rerank_top_n: int = 18
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_query_planning_enabled: bool = True
    rag_max_sub_queries: int = 3
    rag_max_keywords: int = 8
    rag_index_chunk_size: int = 900
    rag_index_chunk_overlap: int = 120
    mcp_enabled: bool = False
    mcp_config_path: Path | None = None
    mcp_tool_prefix: bool = True
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:1357", "http://localhost:1357")
    cors_origin_regexes: tuple[str, ...] = ()
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FalcoSettings":
        path = Path(config_path).resolve()
        config_dir = path.parent
        raw = _resolve_env_refs(_load_yaml(path))

        config_version = _as_int(raw.get("config_version"), default=1)
        model_cfg = _as_dict(raw.get("model"))
        workspace_cfg = _as_dict(raw.get("workspace"))
        agent_cfg = _as_dict(raw.get("agent"))
        controls_cfg = _as_dict(raw.get("controls"))
        memory_cfg = _as_dict(raw.get("memory"))
        rag_cfg = _as_dict(raw.get("rag"))
        mcp_cfg = _as_dict(raw.get("mcp"))
        service_cfg = _as_dict(raw.get("service"))
        skills_cfg = _as_dict(raw.get("skills"))
        secrets_cfg = _as_dict(raw.get("secrets"))

        workspace_root = _resolve_path(config_dir, _as_str(workspace_cfg.get("root"), default="."))

        allowed_raw = workspace_cfg.get("allowed_roots") or ["."]
        if not isinstance(allowed_raw, list):
            raise ValueError("`workspace.allowed_roots` must be a list in config.yaml.")
        allowed_workspace_roots = tuple(
            _resolve_path(config_dir, item) for item in allowed_raw
        ) or (workspace_root,)

        default_working_directory = _resolve_path(
            config_dir,
            _as_str(workspace_cfg.get("default_working_directory"), default="."),
        )
        uploads_root = _resolve_path(
            config_dir,
            _as_str(workspace_cfg.get("uploads_root"), default=str(Path(".falco") / "uploads")),
        )
        runtime_root = _resolve_path(
            config_dir,
            _as_str(workspace_cfg.get("runtime_root"), default=str(Path(".falco") / "runtime")),
        )
        deliverables_root = _resolve_path(
            config_dir,
            _as_str(workspace_cfg.get("deliverables_root"), default=str(Path(".falco") / "deliverables")),
        )
        thread_scoped_directories = _as_bool(workspace_cfg.get("thread_scoped_directories"), default=True)

        memory_root = _resolve_path(
            config_dir,
            _as_str(memory_cfg.get("root"), default=str(Path(".falco") / "memory")),
        )
        skills_public_root = _resolve_path(
            config_dir,
            _as_str(skills_cfg.get("public_root"), default="skills"),
        )
        soul_path_raw = _as_str(agent_cfg.get("soul_path"), default="").strip()
        soul_path = _resolve_path(config_dir, soul_path_raw) if soul_path_raw else None
        skills_user_raw = skills_cfg.get("user_roots") or [str(Path(".falco") / "skills")]
        if not isinstance(skills_user_raw, list):
            raise ValueError("`skills.user_roots` must be a list in config.yaml.")
        skills_user_roots = tuple(_resolve_path(config_dir, item) for item in skills_user_raw)
        mcp_config_path = _resolve_path(
            config_dir,
            _as_str(mcp_cfg.get("config_path"), default=str(Path(".falco") / "mcp.json")),
        )
        rag_milvus_uri = _resolve_path_string(
            config_dir,
            _as_str(
                rag_cfg.get("milvus_uri"),
                default=str(Path(".falco") / "milvus" / "falco_rag.db"),
            ),
        )

        api_key_env = _as_str(secrets_cfg.get("llm_api_key_env"), default="LLM_API_KEY")
        rag_token_env = _as_str(secrets_cfg.get("rag_milvus_token_env"), default="FALCO_RAG_MILVUS_TOKEN")

        api_key = os.getenv(api_key_env, "")
        rag_milvus_token = os.getenv(rag_token_env) or None
        cors_origins_raw = service_cfg.get("cors_origins") or ["http://127.0.0.1:1357", "http://localhost:1357"]
        if not isinstance(cors_origins_raw, list):
            raise ValueError("`service.cors_origins` must be a list in config.yaml.")
        cors_origin_regexes_raw = service_cfg.get("cors_origin_regexes") or []
        if not isinstance(cors_origin_regexes_raw, list):
            raise ValueError("`service.cors_origin_regexes` must be a list in config.yaml.")

        settings = cls(
            config_version=config_version,
            model=_as_str(model_cfg.get("id"), default="gpt-4o-mini"),
            api_key=api_key,
            base_url=_as_str(model_cfg.get("base_url"), default="") or None,
            workspace_root=workspace_root,
            allowed_workspace_roots=allowed_workspace_roots,
            default_working_directory=default_working_directory,
            uploads_root=uploads_root,
            runtime_root=runtime_root,
            deliverables_root=deliverables_root,
            thread_scoped_directories=thread_scoped_directories,
            memory_root=memory_root,
            skills_public_root=skills_public_root,
            skills_user_roots=skills_user_roots,
            soul_path=soul_path,
            max_context_messages=_as_int(agent_cfg.get("max_context_messages"), default=12),
            max_tool_steps=_as_int(agent_cfg.get("max_tool_steps"), default=6),
            max_subagents=_as_int(agent_cfg.get("max_subagents"), default=4),
            subagent_max_steps=_as_int(agent_cfg.get("subagent_max_steps"), default=4),
            tool_read_max_tokens=_as_int(controls_cfg.get("tool_read_max_tokens"), default=100_000),
            tool_write_max_tokens=_as_int(controls_cfg.get("tool_write_max_tokens"), default=200_000),
            tool_search_file_max_bytes=_as_int(controls_cfg.get("tool_search_file_max_bytes"), default=1_000_000),
            tool_search_max_files=_as_int(controls_cfg.get("tool_search_max_files"), default=800),
            tool_list_max_items=_as_int(controls_cfg.get("tool_list_max_items"), default=300),
            blocked_path_parts=_as_str_list(
                controls_cfg.get("blocked_path_parts"),
                default=[".git", ".hg", ".svn", ".falco", "__pycache__", "node_modules", ".next", ".venv", "venv"],
            ),
            blocked_file_names=_as_str_list(
                controls_cfg.get("blocked_file_names"),
                default=[".env", ".env.local", ".env.production", ".env.development", "id_rsa", "id_ed25519"],
            ),
            blocked_file_suffixes=_as_str_list(
                controls_cfg.get("blocked_file_suffixes"),
                default=[".pem", ".key", ".p12", ".pfx"],
            ),
            memory_recent_rounds=_as_int(memory_cfg.get("recent_rounds"), default=10),
            memory_key_rounds=_as_int(memory_cfg.get("key_rounds"), default=10),
            memory_importance_threshold=_as_int(memory_cfg.get("importance_threshold"), default=7),
            memory_max_rounds=_as_int(memory_cfg.get("max_rounds"), default=30),
            memory_context_soft_limit_tokens=_as_int(memory_cfg.get("context_soft_limit_tokens"), default=8192),
            memory_context_max_tokens=_as_int(memory_cfg.get("context_max_tokens"), default=10000),
            memory_silent_turn_cooldown_rounds=_as_int(
                memory_cfg.get("silent_turn_cooldown_rounds"),
                default=4,
            ),
            memory_daily_half_life_days=_as_int(memory_cfg.get("daily_half_life_days"), default=30),
            memory_daily_lookback_days=_as_int(memory_cfg.get("daily_lookback_days"), default=180),
            memory_daily_retrieval_items=_as_int(memory_cfg.get("daily_retrieval_items"), default=8),
            memory_evergreen_retrieval_items=_as_int(memory_cfg.get("evergreen_retrieval_items"), default=5),
            rag_enabled=_as_bool(rag_cfg.get("enabled"), default=True),
            rag_milvus_uri=rag_milvus_uri,
            rag_milvus_token=rag_milvus_token,
            rag_collection=_as_str(rag_cfg.get("collection"), default="falco_knowledge"),
            rag_embedding_model=_as_str(
                rag_cfg.get("embedding_model"),
                default="text-embedding-3-small",
            ),
            rag_base_url=_as_str(rag_cfg.get("base_url"), default="") or None,
            rag_api_key=_as_str(rag_cfg.get("api_key"), default="") or None,
            rag_retrieval_mode=_as_str(rag_cfg.get("retrieval_mode"), default="dense"),
            rag_hybrid_dense_weight=float(rag_cfg.get("hybrid_dense_weight", 0.7)),
            rag_hybrid_sparse_weight=float(rag_cfg.get("hybrid_sparse_weight", 0.3)),
            rag_top_k=_as_int(rag_cfg.get("top_k"), default=5),
            rag_fetch_k=_as_int(rag_cfg.get("fetch_k"), default=18),
            rag_rerank_enabled=_as_bool(rag_cfg.get("rerank_enabled"), default=True),
            rag_rerank_top_n=_as_int(rag_cfg.get("rerank_top_n"), default=18),
            rag_reranker_model=_as_str(
                rag_cfg.get("reranker_model"),
                default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            rag_query_planning_enabled=_as_bool(rag_cfg.get("query_planning_enabled"), default=True),
            rag_max_sub_queries=_as_int(rag_cfg.get("max_sub_queries"), default=3),
            rag_max_keywords=_as_int(rag_cfg.get("max_keywords"), default=8),
            rag_index_chunk_size=_as_int(rag_cfg.get("index_chunk_size"), default=900),
            rag_index_chunk_overlap=_as_int(rag_cfg.get("index_chunk_overlap"), default=120),
            mcp_enabled=_as_bool(mcp_cfg.get("enabled"), default=False),
            mcp_config_path=mcp_config_path,
            mcp_tool_prefix=_as_bool(mcp_cfg.get("tool_prefix"), default=True),
            cors_origins=tuple(str(item).strip() for item in cors_origins_raw if str(item).strip()),
            cors_origin_regexes=tuple(str(item).strip() for item in cors_origin_regexes_raw if str(item).strip()),
            config_path=path,
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.config_version != 2:
            raise ValueError(f"Unsupported config_version={self.config_version}. Expected 2.")
        if not self.allowed_workspace_roots:
            raise ValueError("At least one allowed workspace root is required.")
        for path, name in (
            (self.default_working_directory, "default_working_directory"),
            (self.uploads_root, "uploads_root"),
            (self.runtime_root, "runtime_root"),
            (self.deliverables_root, "deliverables_root"),
        ):
            for root in self.allowed_workspace_roots:
                try:
                    path.relative_to(root)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"{name} must stay within allowed_workspace_roots.")
        if self.max_subagents < 1:
            raise ValueError("agent.max_subagents must be at least 1.")
        if self.subagent_max_steps < 1:
            raise ValueError("agent.subagent_max_steps must be at least 1.")
        if self.tool_read_max_tokens < 1:
            raise ValueError("controls.tool_read_max_tokens must be at least 1.")
        if self.tool_write_max_tokens < 1:
            raise ValueError("controls.tool_write_max_tokens must be at least 1.")
        if self.tool_search_file_max_bytes < 1:
            raise ValueError("controls.tool_search_file_max_bytes must be at least 1.")
        if self.tool_search_max_files < 1:
            raise ValueError("controls.tool_search_max_files must be at least 1.")
        if self.tool_list_max_items < 1:
            raise ValueError("controls.tool_list_max_items must be at least 1.")
        if not self.skills_public_root:
            raise ValueError("skills.public_root is required.")
        if self.soul_path is not None and self.soul_path.suffix.lower() != ".md":
            raise ValueError("agent.soul_path must point to a .md file.")
        if self.rag_retrieval_mode not in {"dense", "hybrid"}:
            raise ValueError("rag.retrieval_mode must be 'dense' or 'hybrid'.")
        if self.rag_rerank_top_n < self.rag_top_k:
            raise ValueError("rag.rerank_top_n must be greater than or equal to rag.top_k.")
        if not self.cors_origins and not self.cors_origin_regexes:
            raise ValueError("At least one service.cors_origins or service.cors_origin_regexes entry is required.")
