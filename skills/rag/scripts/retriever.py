from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalBatch:
    docs: list[Document]
    mode_used: str
    warnings: list[str]


class CandidateRetriever:
    def __init__(
        self,
        *,
        vectorstore_getter,
        retrieval_mode: str,
        fetch_k: int,
        hybrid_dense_weight: float,
        hybrid_sparse_weight: float,
    ) -> None:
        self._vectorstore_getter = vectorstore_getter
        self._retrieval_mode = retrieval_mode
        self._fetch_k = max(1, int(fetch_k))
        self._hybrid_dense_weight = float(hybrid_dense_weight)
        self._hybrid_sparse_weight = float(hybrid_sparse_weight)

    def retrieve(self, query_plan) -> RetrievalBatch:
        vectorstore, hybrid_available, warning = self._vectorstore_getter()
        warnings: list[str] = []
        if warning:
            warnings.append(warning)

        mode_used = "dense"
        if self._retrieval_mode == "hybrid" and hybrid_available:
            mode_used = "hybrid"
        elif self._retrieval_mode == "hybrid" and not hybrid_available:
            warnings.append("Hybrid retrieval was requested but is unavailable; falling back to dense retrieval.")

        queries: list[str] = [query_plan.rewritten_query]
        queries.extend(query_plan.sub_queries)
        if query_plan.keywords:
            queries.append(" ".join(query_plan.keywords))

        unique: dict[tuple[str, str], Document] = {}
        for q in queries:
            q = str(q or "").strip()
            if not q:
                continue
            try:
                if mode_used == "hybrid":
                    docs = vectorstore.similarity_search(
                        q,
                        k=self._fetch_k,
                        ranker_type="weighted",
                        ranker_params={"weights": [self._hybrid_dense_weight, self._hybrid_sparse_weight]},
                    )
                else:
                    docs = vectorstore.similarity_search(q, k=self._fetch_k)
            except Exception:
                docs = []
            for doc in docs:
                key = (str(doc.metadata.get("source", "")), str(doc.metadata.get("chunk_id", "")))
                unique[key] = doc

        return RetrievalBatch(docs=list(unique.values()), mode_used=mode_used, warnings=warnings)

