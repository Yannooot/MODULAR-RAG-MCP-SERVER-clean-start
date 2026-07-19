from typing import Any

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Any]] = []

    def encode(
        self, chunks: list[Chunk], trace: Any | None = None
    ) -> list[ChunkRecord]:
        self.calls.append(([chunk.id for chunk in chunks], trace))
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[float(chunk.metadata["chunk_index"]), 1.0],
            )
            for chunk in chunks
        ]


class FakeSparseEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Any]] = []

    def encode(
        self, chunks: list[Chunk], trace: Any | None = None
    ) -> list[ChunkRecord]:
        self.calls.append(([chunk.id for chunk in chunks], trace))
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                sparse_vector={chunk.text: 1.0},
            )
            for chunk in chunks
        ]


def chunks(count: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{index}",
            text=f"text-{index}",
            metadata={"source_path": "docs/rag.md", "chunk_index": index},
            start_offset=index * 10,
            end_offset=index * 10 + 6,
            source_ref="doc-1",
        )
        for index in range(count)
    ]


def processor(
    batch_size: int = 2,
) -> tuple[BatchProcessor, FakeDenseEncoder, FakeSparseEncoder]:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    instance = BatchProcessor(
        load_settings(),
        batch_size=batch_size,
        dense_encoder=dense,
        sparse_encoder=sparse,
    )
    return instance, dense, sparse


@pytest.mark.unit
def test_five_chunks_are_processed_as_three_stable_batches() -> None:
    batch_processor, dense, sparse = processor(batch_size=2)

    records = batch_processor.process(chunks(5))

    expected_calls = [(["chunk-0", "chunk-1"], None), (["chunk-2", "chunk-3"], None), (["chunk-4"], None)]
    assert dense.calls == expected_calls
    assert sparse.calls == expected_calls
    assert [record.id for record in records] == [f"chunk-{index}" for index in range(5)]
    assert [record.dense_vector for record in records] == [
        [float(index), 1.0] for index in range(5)
    ]
    assert [record.sparse_vector for record in records] == [
        {f"text-{index}": 1.0} for index in range(5)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_size_must_be_a_positive_integer(batch_size: Any) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BatchProcessor(load_settings(), batch_size=batch_size)


@pytest.mark.unit
def test_empty_input_does_not_call_encoders() -> None:
    batch_processor, dense, sparse = processor()

    assert batch_processor.process([]) == []
    assert dense.calls == []
    assert sparse.calls == []


@pytest.mark.unit
def test_trace_is_forwarded_and_records_each_batch_duration() -> None:
    batch_processor, dense, sparse = processor(batch_size=2)
    trace = TraceContext()

    batch_processor.process(chunks(5), trace)

    assert all(call[1] is trace for call in dense.calls + sparse.calls)
    batch_stages = [
        stage for stage in trace.stages if stage["name"] == "embedding_batch"
    ]
    assert [stage["details"]["chunk_count"] for stage in batch_stages] == [2, 2, 1]
    assert [stage["details"]["batch_index"] for stage in batch_stages] == [0, 1, 2]
    assert all(stage["details"]["elapsed_ms"] >= 0 for stage in batch_stages)
    assert trace.stages[-1]["name"] == "batch_processor"
    assert trace.stages[-1]["details"]["batch_count"] == 3
    assert trace.stages[-1]["details"]["chunk_count"] == 5
    assert trace.stages[-1]["details"]["batch_size"] == 2


@pytest.mark.unit
def test_encoder_result_count_must_match_batch() -> None:
    batch_processor, dense, _ = processor()
    dense.encode = lambda batch, trace=None: []  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="count"):
        batch_processor.process(chunks(1))


@pytest.mark.unit
def test_encoder_record_ids_must_align() -> None:
    batch_processor, _, sparse = processor()

    def wrong_ids(batch: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                id="wrong-id",
                text=batch[0].text,
                metadata=dict(batch[0].metadata),
                sparse_vector={"term": 1.0},
            )
        ]

    sparse.encode = wrong_ids  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="IDs"):
        batch_processor.process(chunks(1))


@pytest.mark.unit
def test_required_dense_and_sparse_vectors_must_exist() -> None:
    batch_processor, _, sparse = processor()

    def missing_vector(
        batch: list[Chunk], trace: Any | None = None
    ) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                id=batch[0].id,
                text=batch[0].text,
                metadata=dict(batch[0].metadata),
            )
        ]

    sparse.encode = missing_vector  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="sparse vector"):
        batch_processor.process(chunks(1))
