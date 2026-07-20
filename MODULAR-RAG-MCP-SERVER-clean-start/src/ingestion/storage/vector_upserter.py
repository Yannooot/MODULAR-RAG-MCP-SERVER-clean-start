"""Convert dense chunk records into the vector store write contract."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from copy import deepcopy
from numbers import Real
from typing import TYPE_CHECKING, Any

from core.settings import Settings
from core.types import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class VectorUpserter:
    def __init__(
        self,
        settings: Settings,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def upsert(
        self,
        records: Sequence[ChunkRecord],
        trace: TraceContext | None = None,
    ) -> list[str]:
        payloads = [self._to_payload(record) for record in records]
        if not payloads:
            return []
        self.vector_store.upsert(payloads, trace)
        return [payload["id"] for payload in payloads]

    @classmethod
    def generate_chunk_id(cls, record: ChunkRecord) -> str:
        source_path, chunk_index = cls._identity_fields(record)
        content_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        identity = f"{source_path}{chunk_index}{content_hash[:8]}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @classmethod
    def _to_payload(cls, record: ChunkRecord) -> dict[str, Any]:
        if not isinstance(record, ChunkRecord):
            raise ValueError("records must contain ChunkRecord values")
        cls._identity_fields(record)
        vector = record.dense_vector
        if (
            not isinstance(vector, list)
            or not vector
            or any(
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError("dense_vector must be non-empty and finite")
        return {
            "id": cls.generate_chunk_id(record),
            "text": record.text,
            "metadata": deepcopy(record.metadata),
            "vector": [float(value) for value in vector],
        }

    @staticmethod
    def _identity_fields(record: ChunkRecord) -> tuple[str, int]:
        if not isinstance(record, ChunkRecord):
            raise ValueError("records must contain ChunkRecord values")
        source_path = record.metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("metadata.source_path must be a non-empty string")
        chunk_index = record.metadata.get("chunk_index")
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
        ):
            raise ValueError("metadata.chunk_index must be a non-negative integer")
        return source_path, chunk_index
