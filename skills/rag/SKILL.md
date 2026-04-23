---
name: rag
description: Local knowledge retrieval and indexing skill.
enabled: true
---
Use this skill when the task needs local knowledge-base evidence or indexing.

Actions available through `use_skill`:
- search: args={"query": "...", "top_k": 5}
- index: args={"path": "knowledge", "drop_old": false}
- refresh_source: reserved for future incremental refresh support
- remove_source: reserved for future source-removal support
- status: reserved for future index-status inspection

Prefer search before answering questions that depend on local documents.
Indexing and other maintenance actions change the knowledge base and require human approval.

