from __future__ import annotations

import argparse
from pathlib import Path

from harness.config.config import FalcoSettings
from harness.rag import MilvusRAG
from langchain_openai import ChatOpenAI


def _resolve_workspace_path(root: Path, raw_path: str) -> Path:
    target = (root / raw_path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Path escapes workspace root: {raw_path}") from exc
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Falco RAG indexing and search helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("index", help="Index local files into Milvus.")
    build_parser.add_argument("--path", default="knowledge", help="Workspace-relative path to index.")
    build_parser.add_argument("--drop-old", action="store_true", help="Drop existing collection data first.")

    search_parser = sub.add_parser("search", help="Search local Milvus knowledge base.")
    search_parser.add_argument("--query", required=True, help="Query text.")
    search_parser.add_argument("--top-k", type=int, default=5, help="Top k documents after rerank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = FalcoSettings.from_yaml(Path(__file__).resolve().parents[1] / "config.yaml")
    llm = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )
    rag = MilvusRAG(settings=settings, llm=llm)

    if args.command == "index":
        target = _resolve_workspace_path(Path(settings.workspace_root), args.path)
        print(rag.index_paths([target], drop_old=args.drop_old))
        return

    if args.command == "search":
        result = rag.search(args.query, top_k=args.top_k)
        print(result.render())


if __name__ == "__main__":
    main()
