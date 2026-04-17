# Falco RAG (LangChain + Milvus + Hybrid + Query Optimization + Rerank)

## Features implemented

- Milvus vector database integration via `langchain-milvus`
- Hybrid retrieval support (dense + sparse/BM25 when Milvus server supports built-in sparse)
- Query construction and optimization (LLM-based rewrite + sub-query expansion + keyword extraction)
- Cross-encoder reranking (`sentence-transformers` cross encoder)
- Connected into Falco as the enabled `rag` skill:
  - `use_skill(skill_name="rag", action="search", args={"query": "...", "top_k": 5})`
  - `use_skill(skill_name="rag", action="index", args={"path": "knowledge", "drop_old": false})`
- API endpoints:
  - `POST /api/v1/rag/search`
  - `POST /api/v1/rag/index`

## Environment variables

```bash
FALCO_RAG_ENABLED=true
FALCO_RAG_MILVUS_URI=./.falco/milvus/falco_rag.db
FALCO_RAG_MILVUS_TOKEN=
FALCO_RAG_COLLECTION=falco_knowledge
FALCO_RAG_EMBEDDING_MODEL=text-embedding-3-small
FALCO_RAG_TOP_K=5
FALCO_RAG_FETCH_K=18
FALCO_RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Index local knowledge

Use CLI:

```bash
python -m harness.rag_cli index --path knowledge --drop-old
```

Or via agent skill:

- Ask Falco to use the `rag` skill with action `index`.
- Indexing updates the local knowledge base and requires human approval.

Or via API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/index \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"knowledge\", \"drop_old\": true}"
```

## Search

CLI:

```bash
python -m harness.rag_cli search --query "memory design" --top-k 5
```

Agent skill:

```text
use_skill(skill_name="rag", action="search", args={"query": "memory design", "top_k": 5})
```

API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"memory design\", \"top_k\":5}"
```

## Important note on hybrid retrieval

- If `FALCO_RAG_MILVUS_URI` points to local `*.db` (Milvus Lite), implementation will automatically degrade to dense retrieval.
- Full hybrid sparse+dense requires Milvus server mode that supports built-in BM25 sparse function.
