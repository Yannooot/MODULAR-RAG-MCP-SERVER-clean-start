"""Coordinate dense and sparse encoders in stable batches."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from time import perf_counter

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    def __init__(
        self,
        settings: Settings,
        batch_size: int = 32,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        self.settings = settings
        self.batch_size = batch_size
        self.dense_encoder = dense_encoder or DenseEncoder(settings)
        self.sparse_encoder = sparse_encoder or SparseEncoder(settings)

    def process(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        chunk_list = list(chunks)
        records: list[ChunkRecord] = []
        total_started = perf_counter()

        for batch_index, start in enumerate(
            range(0, len(chunk_list), self.batch_size)
        ):
            batch = chunk_list[start : start + self.batch_size]
            batch_started = perf_counter()
            dense_records = self.dense_encoder.encode(batch, trace)
            sparse_records = self.sparse_encoder.encode(batch, trace)
            records.extend(self._merge_batch(batch, dense_records, sparse_records))
            if trace is not None:
                trace.record_stage(
                    "embedding_batch",
                    {
                        "batch_index": batch_index,
                        "chunk_count": len(batch),
                        "elapsed_ms": (perf_counter() - batch_started) * 1000,
                    },
                )

        if trace is not None:
            trace.record_stage(
                "batch_processor",
                {
                    "chunk_count": len(chunk_list),
                    "batch_count": (
                        len(chunk_list) + self.batch_size - 1
                    )
                    // self.batch_size,
                    "batch_size": self.batch_size,
                    "elapsed_ms": (perf_counter() - total_started) * 1000,
                },
            )
        return records

    @staticmethod
    def _merge_batch(
        chunks: Sequence[Chunk],
        dense_records: Sequence[ChunkRecord],
        sparse_records: Sequence[ChunkRecord],
    ) -> list[ChunkRecord]:
        if len(dense_records) != len(chunks) or len(sparse_records) != len(chunks):
            raise ValueError("encoder result count must match batch chunk count")

        merged: list[ChunkRecord] = []
        for chunk, dense, sparse in zip(
            chunks, dense_records, sparse_records, strict=True
        ):
            if dense.id != chunk.id or sparse.id != chunk.id:
                raise ValueError("encoder record IDs must align with batch chunk IDs")
            if dense.dense_vector is None:
                raise ValueError(f"dense vector is missing for chunk {chunk.id}")
            if sparse.sparse_vector is None:
                raise ValueError(f"sparse vector is missing for chunk {chunk.id}")
            merged.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=deepcopy(dense.metadata),
                    dense_vector=list(dense.dense_vector),
                    sparse_vector=dict(sparse.sparse_vector),
                )
            )
        return merged
