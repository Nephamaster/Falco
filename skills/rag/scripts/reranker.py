from __future__ import annotations

from harness.tokenization import truncate_tokens


class CrossEncoderReranker:
    def __init__(self, *, model_name: str, enabled: bool, top_n: int) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._top_n = max(1, int(top_n))
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(self._model_name)
        return self._reranker

    def rerank(self, query: str, docs: list, *, final_top_k: int) -> list:
        if not docs:
            return []
        top_n = min(self._top_n, len(docs))
        if not self._enabled:
            return docs[: min(final_top_k, top_n)]
        try:
            reranker = self._get_reranker()
            working_set = docs[:top_n]
            pairs = [(query, truncate_tokens(doc.page_content, 2200)) for doc in working_set]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(working_set, scores, strict=False), key=lambda item: float(item[1]), reverse=True)
            return [doc for doc, _ in ranked[:final_top_k]]
        except Exception:
            return docs[: min(final_top_k, top_n)]

