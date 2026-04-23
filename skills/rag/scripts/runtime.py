from __future__ import annotations

from pathlib import Path

from skills.rag.scripts.renderer import render_search_result
from skills.rag.scripts.service import RAGService

_SERVICE_CACHE: dict[str, RAGService] = {}


def _service_key(context) -> str:
    return f"{context.settings.rag_collection}|{context.settings.rag_milvus_uri}|{context.settings.rag_retrieval_mode}"


def _get_service(context) -> RAGService:
    key = _service_key(context)
    service = _SERVICE_CACHE.get(key)
    if service is None:
        service = RAGService(settings=context.settings, llm=context.llm)
        _SERVICE_CACHE[key] = service
    return service


def execute(*, action: str, args: dict, context) -> str:
    if not getattr(context.settings, "rag_enabled", True):
        return "RAG skill is disabled."
    service = _get_service(context)
    normalized_action = str(action or "").strip().lower()
    if normalized_action == "search":
        query = str(args.get("query", "")).strip()
        top_k = int(args.get("top_k", context.settings.rag_top_k))
        if not query:
            return "RAG search requires args.query."
        return render_search_result(service.search(query, top_k=top_k))
    if normalized_action == "index":
        raw_path = str(args.get("path", "knowledge")).strip()
        drop_old = bool(args.get("drop_old", False))
        if context.workspace is None:
            return "RAG indexing requires workspace context."
        target = context.workspace.resolve_thread_path(
            context.thread_id,
            raw_path,
            cwd=context.working_directory,
        )
        if not target.exists():
            return f"Path does not exist: {raw_path}"
        return service.index_paths([Path(target)], drop_old=drop_old)
    if normalized_action == "refresh_source":
        return service.refresh_source(args)
    if normalized_action == "remove_source":
        return service.remove_source(args)
    if normalized_action == "status":
        return service.status()
    return "RAG skill supports actions: search, index, refresh_source, remove_source, status."
