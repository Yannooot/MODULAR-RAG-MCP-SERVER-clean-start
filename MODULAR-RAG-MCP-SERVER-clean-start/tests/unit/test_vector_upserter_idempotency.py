import hashlib
from copy import deepcopy
from typing import Any

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from ingestion.storage.vector_upserter import VectorUpserter


class MemoryVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[list[dict[str, Any]], TraceContext | None]] = []

    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: TraceContext | None = None,
    ) -> None:
        copied = deepcopy(records)
        self.calls.append((copied, trace))
        for record in copied:
            self.records[record["id"]] = record


def record(
    text: str = "same content",
    *,
    source_path: str = "docs/sample.md",
    chunk_index: int = 2,
    vector: list[float] | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        id="encoder-id",
        text=text,
        metadata={
            "source_path": source_path,
            "chunk_index": chunk_index,
            "title": "Sample",
        },
        dense_vector=[0.1, 0.2] if vector is None else vector,
    )


@pytest.mark.unit
def test_repeated_upsert_uses_same_id_without_duplicate_records() -> None:
    store = MemoryVectorStore()
    upserter = VectorUpserter(load_settings(), store)

    first = upserter.upsert([record()])
    second = upserter.upsert([record()])

    assert first == second
    assert len(store.records) == 1


@pytest.mark.unit
def test_id_matches_spec_formula_and_changes_with_content() -> None:
    store = MemoryVectorStore()
    upserter = VectorUpserter(load_settings(), store)
    original = record()
    changed = record("changed content")

    identifiers = upserter.upsert([original, changed])
    content_hash = hashlib.sha256(original.text.encode("utf-8")).hexdigest()
    expected = hashlib.sha256(
        f"docs/sample.md2{content_hash[:8]}".encode("utf-8")
    ).hexdigest()

    assert identifiers[0] == expected
    assert identifiers[0] != identifiers[1]


@pytest.mark.unit
def test_batch_upsert_preserves_order_and_complete_payload() -> None:
    store = MemoryVectorStore()
    upserter = VectorUpserter(load_settings(), store)
    records = [
        record("first", chunk_index=0, vector=[1.0, 0.0]),
        record("second", chunk_index=1, vector=[0.0, 1.0]),
    ]

    identifiers = upserter.upsert(records)
    written, _ = store.calls[0]

    assert [item["id"] for item in written] == identifiers
    assert [item["text"] for item in written] == ["first", "second"]
    assert [item["vector"] for item in written] == [[1.0, 0.0], [0.0, 1.0]]
    assert written[0]["metadata"] == records[0].metadata
    assert written[0]["metadata"] is not records[0].metadata


@pytest.mark.unit
def test_trace_is_forwarded_and_empty_input_skips_store() -> None:
    store = MemoryVectorStore()
    upserter = VectorUpserter(load_settings(), store)
    trace = TraceContext()

    assert upserter.upsert([], trace) == []
    identifiers = upserter.upsert([record()], trace)

    assert len(identifiers) == 1
    assert len(store.calls) == 1
    assert store.calls[0][1] is trace


@pytest.mark.unit
@pytest.mark.parametrize("vector", [None, [], [float("nan")], [float("inf")]])
def test_dense_vector_must_be_present_and_finite(
    vector: list[float] | None,
) -> None:
    candidate = record()
    candidate.dense_vector = vector

    with pytest.raises(ValueError, match="dense_vector"):
        VectorUpserter(load_settings(), MemoryVectorStore()).upsert([candidate])


@pytest.mark.unit
@pytest.mark.parametrize("chunk_index", [None, -1, True])
def test_chunk_index_must_be_a_non_negative_integer(chunk_index: Any) -> None:
    candidate = record()
    candidate.metadata["chunk_index"] = chunk_index

    with pytest.raises(ValueError, match="chunk_index"):
        VectorUpserter(load_settings(), MemoryVectorStore()).upsert([candidate])
