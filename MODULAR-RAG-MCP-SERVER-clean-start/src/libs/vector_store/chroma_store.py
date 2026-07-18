"""Persistent Chroma vector store adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class ChromaStore(BaseVectorStore):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        persist_path = settings.vector_store.persist_path
        collection_name = settings.vector_store.collection_name
        if not isinstance(persist_path, str) or not persist_path.strip():
            raise ValueError("vector_store.persist_path must be configured")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("vector_store.collection_name must be configured")

        self._client = chromadb.PersistentClient(
            path=str(Path(persist_path).expanduser())
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name.strip(), embedding_function=None
        )

    def upsert(
        self, records: list[dict[str, Any]], trace: TraceContext | None = None
    ) -> None:
        if not isinstance(records, list):
            raise TypeError("records must be a list")
        if not records:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any] | None] = []
        embeddings: list[list[float]] = []
        for index, record in enumerate(records):
            record_id, text, metadata, vector = self._validate_record(record, index)
            ids.append(record_id)
            documents.append(text)
            metadatas.append(metadata or None)
            embeddings.append(vector)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self._validate_vector(vector, "query vector")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None and not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping")
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=dict(filters) if filters else None,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_query_result(result)

    def _validate_record(
        self, record: dict[str, Any], index: int
    ) -> tuple[str, str, dict[str, Any], list[float]]:
        if not isinstance(record, dict):
            raise TypeError(f"record {index} must be a mapping")
        record_id = record.get("id")
        text = record.get("text")
        metadata = record.get("metadata", {})
        vector = record.get("vector", record.get("dense_vector"))
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index} requires a non-empty id")
        if not isinstance(text, str):
            raise ValueError(f"record {index} requires string text")
        if not isinstance(metadata, dict):
            raise ValueError(f"record {index} metadata must be a mapping")
        return record_id, text, metadata, self._validate_vector(
            vector, f"record {index} vector"
        )

    @staticmethod
    def _validate_vector(vector: Any, field_name: str) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"{field_name} must be a non-empty list")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in vector
        ):
            raise ValueError(f"{field_name} values must be numeric")
        return [float(value) for value in vector]

    @staticmethod
    def _format_query_result(result: Any) -> list[dict[str, Any]]:
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "id": record_id,
                "text": document or "",
                "metadata": metadata or {},
                "score": 1.0 / (1.0 + max(float(distance), 0.0)),
            }
            for record_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
