from __future__ import annotations

from harness.tokenization import truncate_tokens


def render_search_result(result, *, max_tokens_per_doc: int = 700) -> str:
    if not result.reranked_docs:
        return "No knowledge found in local vector database."
    lines = [
        f"Optimized query: {result.query_plan.rewritten_query}",
        f"Query variants: {', '.join(result.query_plan.sub_queries) if result.query_plan.sub_queries else '(none)'}",
        f"Retrieval mode: {result.retrieval_mode}",
    ]
    if result.warnings:
        lines.append(f"Warnings: {' | '.join(result.warnings)}")
    lines.extend(["", "Retrieved context:"])
    for idx, doc in enumerate(result.reranked_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "n/a")
        text = doc.page_content.strip().replace("\n", " ")
        lines.append(f"[{idx}] source={source} chunk={chunk_id}")
        lines.append(truncate_tokens(text, max_tokens_per_doc))
        lines.append("")
    return "\n".join(lines).strip()

