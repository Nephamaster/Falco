from __future__ import annotations

from skills.rag.scripts.models import QueryPlan


class QueryPlanner:
    def __init__(self, *, llm, prompt: str, enabled: bool, max_sub_queries: int, max_keywords: int) -> None:
        self._llm = llm
        self._prompt = prompt
        self._enabled = enabled
        self._max_sub_queries = max(0, int(max_sub_queries))
        self._max_keywords = max(0, int(max_keywords))

    def plan(self, query: str) -> QueryPlan:
        cleaned = " ".join(str(query or "").split())
        fallback = QueryPlan(rewritten_query=cleaned, sub_queries=[], keywords=[])
        if not cleaned or self._llm is None or not self._enabled:
            return fallback
        try:
            planner = self._llm.with_structured_output(QueryPlan)
            plan = planner.invoke(
                [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": cleaned},
                ]
            )
            rewritten = str(plan.rewritten_query or "").strip()
            if not rewritten:
                return fallback
            return QueryPlan(
                rewritten_query=rewritten,
                sub_queries=[item.strip() for item in plan.sub_queries if str(item).strip()][: self._max_sub_queries],
                keywords=[item.strip() for item in plan.keywords if str(item).strip()][: self._max_keywords],
            )
        except Exception:
            return fallback

