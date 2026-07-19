from typing import Any

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.sparse_encoder import SparseEncoder, default_tokenize


def chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        metadata={"source_path": "docs/bm25.md", "tags": ["BM25"]},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_english_tokens_are_normalized_to_term_frequencies() -> None:
    record = SparseEncoder(load_settings()).encode(
        [chunk("chunk-1", "Vector search, vector DATABASE!")]
    )[0]

    assert record.sparse_vector == {
        "vector": 2.0,
        "search": 1.0,
        "database": 1.0,
    }


@pytest.mark.unit
def test_chinese_text_uses_overlapping_bigrams() -> None:
    assert default_tokenize("向量数据库") == ["向量", "量数", "数据", "据库"]


@pytest.mark.unit
def test_output_preserves_chunk_contract_and_order() -> None:
    chunks = [chunk("chunk-1", "alpha"), chunk("chunk-2", "beta beta")]

    records = SparseEncoder(load_settings()).encode(chunks)

    assert all(isinstance(record, ChunkRecord) for record in records)
    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert [record.text for record in records] == ["alpha", "beta beta"]
    assert [record.sparse_vector for record in records] == [
        {"alpha": 1.0},
        {"beta": 2.0},
    ]
    assert all(record.dense_vector is None for record in records)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "   ", "，。！？---"])
def test_empty_or_punctuation_only_text_has_empty_sparse_vector(text: str) -> None:
    record = SparseEncoder(load_settings()).encode([chunk("chunk-1", text)])[0]

    assert record.sparse_vector == {}


@pytest.mark.unit
def test_empty_chunk_list_returns_empty_records() -> None:
    assert SparseEncoder(load_settings()).encode([]) == []


@pytest.mark.unit
def test_custom_tokenizer_can_replace_default_strategy() -> None:
    received: list[str] = []

    def tokenizer(text: str) -> list[str]:
        received.append(text)
        return ["RAG", "rag", "检索"]

    record = SparseEncoder(load_settings(), tokenizer=tokenizer).encode(
        [chunk("chunk-1", "custom text")]
    )[0]

    assert received == ["custom text"]
    assert record.sparse_vector == {"rag": 2.0, "检索": 1.0}


@pytest.mark.unit
def test_metadata_is_deep_copied() -> None:
    source = chunk("chunk-1", "bm25")

    record = SparseEncoder(load_settings()).encode([source])[0]
    record.metadata["tags"].append("检索")

    assert source.metadata["tags"] == ["BM25"]


@pytest.mark.unit
def test_trace_records_token_statistics() -> None:
    trace = TraceContext()

    SparseEncoder(load_settings()).encode(
        [chunk("chunk-1", "alpha beta"), chunk("chunk-2", "beta gamma")],
        trace,
    )

    assert trace.stages[-1] == {
        "name": "sparse_encoder",
        "details": {
            "chunk_count": 2,
            "token_count": 4,
            "vocabulary_size": 3,
        },
    }
