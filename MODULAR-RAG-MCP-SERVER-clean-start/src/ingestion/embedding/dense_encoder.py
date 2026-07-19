"""Adapt chunk batches to the configured dense embedding provider."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    def __init__(
        self, settings: Settings, embedding: BaseEmbedding | None = None
    ) -> None:
        self.settings = settings
        self.embedding = embedding or EmbeddingFactory.create(settings)

    def encode(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        if not chunks:
            if trace is not None:
                trace.record_stage(
                    "dense_encoder",
                    {"chunk_count": 0, "vector_dimension": 0},
                )
            return []

        vectors = self.embedding.embed([chunk.text for chunk in chunks], trace)
        dimension = self._validate_vectors(vectors, len(chunks))
        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=deepcopy(chunk.metadata),
                dense_vector=[float(value) for value in vector],
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if trace is not None:
            trace.record_stage(
                "dense_encoder",
                {"chunk_count": len(chunks), "vector_dimension": dimension},
            )
        return records

    @staticmethod
    def _validate_vectors(vectors: object, chunk_count: int) -> int:
        if not isinstance(vectors, list) or len(vectors) != chunk_count:
            raise ValueError("embedding vector count must match chunk count")

        dimension: int | None = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise ValueError("embedding vectors must be non-empty lists")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise ValueError("embedding vectors must contain finite numeric values")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("embedding vectors must use the same dimension")
        return dimension or 0
