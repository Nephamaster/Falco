from __future__ import annotations

import hashlib

from pymilvus import connections


def _preferred_alias(connection_args: dict) -> str:
    uri = str((connection_args or {}).get("uri", "")).strip()
    token = str((connection_args or {}).get("token", "")).strip()
    digest = hashlib.sha1(f"{uri}|{token}".encode("utf-8")).hexdigest()[:12]
    return f"falco_{digest}"


def ensure_milvus_connection(connection_args: dict, alias: str | None = None) -> str:
    alias = alias or _preferred_alias(connection_args)
    try:
        existing = connections.get_connection_addr(alias)
    except Exception:
        existing = None
    if existing:
        return alias
    connections.connect(alias=alias, **dict(connection_args or {}))
    return alias


def _build_connected_milvus_client(base_cls, connection_args: dict):
    class ConnectedMilvusClient(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            alias = getattr(self, "_using", None) or _preferred_alias(connection_args)
            ensure_milvus_connection(connection_args, alias=alias)

    return ConnectedMilvusClient


def _instantiate_milvus(constructor, **kwargs):
    import langchain_milvus.vectorstores.milvus as milvus_module

    connection_args = dict(kwargs.get("connection_args") or {})
    ensure_milvus_connection(connection_args)

    original_client = milvus_module.MilvusClient
    milvus_module.MilvusClient = _build_connected_milvus_client(original_client, connection_args)
    try:
        return constructor(**kwargs)
    finally:
        milvus_module.MilvusClient = original_client


def build_connected_milvus(**kwargs):
    from langchain_milvus import Milvus

    return _instantiate_milvus(Milvus, **kwargs)


def build_connected_milvus_from_documents(**kwargs):
    from langchain_milvus import Milvus

    return _instantiate_milvus(Milvus.from_documents, **kwargs)
