from typing import Any

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[list[str], Any]] = []

    def embed(
        self, texts: list[str], trace: Any | None = None
    ) -> list[list[float]]:
        self.calls.append((texts, trace))
        return self.vectors


def chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        metadata={
            "source_path": "docs/rag.md",
            "tags": ["RAG"],
        },
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_encodes_all_chunk_texts_in_one_batch() -> None:
    embedding = FakeEmbedding([[0.1, 0.2], [0.3, 0.4]])
    chunks = [chunk("chunk-1", "first"), chunk("chunk-2", "second")]

    records = DenseEncoder(load_settings(), embedding=embedding).encode(chunks)

    assert embedding.calls == [(["first", "second"], None)]
    assert all(isinstance(record, ChunkRecord) for record in records)
    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert [record.text for record in records] == ["first", "second"]
    assert [record.dense_vector for record in records] == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert all(record.sparse_vector is None for record in records)


@pytest.mark.unit
def test_uses_embedding_factory_when_provider_is_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = FakeEmbedding([[1.0]])
    settings = load_settings()
    captured: dict[str, Any] = {}

    def create(received_settings: Any) -> FakeEmbedding:
        captured["settings"] = received_settings
        return embedding

    monkeypatch.setattr(EmbeddingFactory, "create", create)

    records = DenseEncoder(settings).encode([chunk("chunk-1", "content")])

    assert captured["settings"] is settings
    assert records[0].dense_vector == [1.0]


@pytest.mark.unit
def test_metadata_is_deep_copied() -> None:
    source = chunk("chunk-1", "content")

    record = DenseEncoder(
        load_settings(), embedding=FakeEmbedding([[1.0, 2.0]])
    ).encode([source])[0]
    record.metadata["tags"].append("向量")

    assert source.metadata["tags"] == ["RAG"]


@pytest.mark.unit
def test_empty_input_returns_without_calling_provider() -> None:
    embedding = FakeEmbedding([])

    records = DenseEncoder(load_settings(), embedding=embedding).encode([])

    assert records == []
    assert embedding.calls == []


@pytest.mark.unit
def test_vector_count_must_match_chunk_count() -> None:
    encoder = DenseEncoder(load_settings(), embedding=FakeEmbedding([[1.0]]))

    with pytest.raises(ValueError, match="count"):
        encoder.encode([chunk("chunk-1", "first"), chunk("chunk-2", "second")])


@pytest.mark.unit
@pytest.mark.parametrize(
    "vectors",
    [
        [[], []],
        [[1.0], [1.0, 2.0]],
        [[1.0, float("nan")], [2.0, 3.0]],
    ],
)
def test_vectors_must_be_non_empty_finite_and_same_dimension(
    vectors: list[list[float]],
) -> None:
    encoder = DenseEncoder(load_settings(), embedding=FakeEmbedding(vectors))

    with pytest.raises(ValueError, match="vectors"):
        encoder.encode([chunk("chunk-1", "first"), chunk("chunk-2", "second")])


@pytest.mark.unit
def test_trace_is_forwarded_and_records_summary() -> None:
    embedding = FakeEmbedding([[1.0, 2.0, 3.0]])
    trace = TraceContext()

    DenseEncoder(load_settings(), embedding=embedding).encode(
        [chunk("chunk-1", "content")], trace
    )

    assert embedding.calls[0][1] is trace
    assert trace.stages[-1] == {
        "name": "dense_encoder",
        "details": {"chunk_count": 1, "vector_dimension": 3},
    }
