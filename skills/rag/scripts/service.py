from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from harness.prompts.templates import RAG_QUERY_OPTIMIZATION_PROMPT_TEMPLATE
from skills.rag.scripts.indexer import KnowledgeIndexer
from skills.rag.scripts.milvus_connection import build_connected_milvus
from skills.rag.scripts.models import SearchArtifacts
from skills.rag.scripts.query_planner import QueryPlanner
from skills.rag.scripts.retriever import CandidateRetriever
from skills.rag.scripts.reranker import CrossEncoderReranker


class RAGService:
    def __init__(self, settings, llm=None) -> None:
        self.settings = settings
        self.llm = llm
        self._vectorstore_state = None
        self._embeddings = OpenAIEmbeddings(
            model=self.settings.rag_embedding_model,
            api_key=getattr(self.settings, "rag_api_key", self.settings.api_key),
            base_url=getattr(self.settings, "rag_base_url", self.settings.base_url),
            check_embedding_ctx_length=False,
        )
        self._query_planner = QueryPlanner(
            llm=llm,
            prompt=RAG_QUERY_OPTIMIZATION_PROMPT_TEMPLATE,
            enabled=self.settings.rag_query_planning_enabled,
            max_sub_queries=self.settings.rag_max_sub_queries,
            max_keywords=self.settings.rag_max_keywords,
        )
        self._reranker = CrossEncoderReranker(
            model_name=self.settings.rag_reranker_model,
            enabled=self.settings.rag_rerank_enabled,
            top_n=self.settings.rag_rerank_top_n,
        )
        self._retriever = CandidateRetriever(
            vectorstore_getter=self._get_vectorstore,
            retrieval_mode=self.settings.rag_retrieval_mode,
            fetch_k=self.settings.rag_fetch_k,
            hybrid_dense_weight=self.settings.rag_hybrid_dense_weight,
            hybrid_sparse_weight=self.settings.rag_hybrid_sparse_weight,
        )
        self._indexer = KnowledgeIndexer(
            embeddings=self._embeddings,
            connection_args=self.connection_args,
            collection_name=self.settings.rag_collection,
            retrieval_mode=self.settings.rag_retrieval_mode,
        )

    @property
    def connection_args(self) -> dict:
        args: dict = {"uri": self.settings.rag_milvus_uri}
        if self.settings.rag_milvus_token:
            args["token"] = self.settings.rag_milvus_token
        return args

    def _get_vectorstore(self):
        if self._vectorstore_state is not None:
            return self._vectorstore_state

        kwargs = {
            "embedding_function": self._embeddings,
            "connection_args": self.connection_args,
            "collection_name": self.settings.rag_collection,
            "drop_old": False,
        }
        hybrid_available = False
        warning = ""
        if self.settings.rag_retrieval_mode == "hybrid":
            try:
                from langchain_milvus import BM25BuiltInFunction

                kwargs["builtin_function"] = BM25BuiltInFunction(output_field_names="sparse")
                kwargs["vector_field"] = ["dense", "sparse"]
                hybrid_available = True
            except Exception:
                hybrid_available = False
                warning = "BM25 hybrid retrieval is unavailable in the current Milvus environment."

        self._vectorstore_state = (build_connected_milvus(**kwargs), hybrid_available, warning)
        return self._vectorstore_state

    def search(self, query: str, top_k: int | None = None) -> SearchArtifacts:
        final_k = top_k if isinstance(top_k, int) and top_k > 0 else self.settings.rag_top_k
        plan = self._query_planner.plan(query)
        retrieval_batch = self._retriever.retrieve(plan)
        reranked = self._reranker.rerank(plan.rewritten_query, retrieval_batch.docs, final_top_k=final_k)
        return SearchArtifacts(
            query_plan=plan,
            candidates=retrieval_batch.docs,
            reranked_docs=reranked,
            retrieval_mode=retrieval_batch.mode_used,
            warnings=retrieval_batch.warnings,
        )

    def index_paths(self, paths: list, *, drop_old: bool = False) -> str:
        self._vectorstore_state = None
        return self._indexer.index_paths(
            paths,
            chunk_size=self.settings.rag_index_chunk_size,
            chunk_overlap=self.settings.rag_index_chunk_overlap,
            drop_old=drop_old,
        )

    def refresh_source(self, *_args, **_kwargs) -> str:
        return "RAG maintenance action `refresh_source` is reserved and not implemented yet."

    def remove_source(self, *_args, **_kwargs) -> str:
        return "RAG maintenance action `remove_source` is reserved and not implemented yet."

    def status(self) -> str:
        return (
            f"collection={self.settings.rag_collection} "
            f"mode={self.settings.rag_retrieval_mode} "
            f"top_k={self.settings.rag_top_k} "
            f"fetch_k={self.settings.rag_fetch_k} "
            f"rerank_top_n={self.settings.rag_rerank_top_n}"
        )
