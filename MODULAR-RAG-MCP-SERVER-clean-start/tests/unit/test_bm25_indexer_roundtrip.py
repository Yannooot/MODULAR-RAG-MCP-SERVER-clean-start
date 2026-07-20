import math
from pathlib import Path

import pytest

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer


def record(
    identifier: str,
    terms: dict[str, float],
    source_path: str = "docs/corpus.md",
) -> ChunkRecord:
    return ChunkRecord(
        id=identifier,
        text=f"text for {identifier}",
        metadata={"source_path": source_path},
        sparse_vector=terms,
    )


def corpus() -> list[ChunkRecord]:
    return [
        record("chunk-a", {"apple": 2.0, "banana": 1.0}),
        record("chunk-b", {"banana": 2.0, "carrot": 1.0}),
        record("chunk-c", {"carrot": 1.0, "durian": 1.0}),
    ]


@pytest.mark.unit
def test_build_load_and_query_return_stable_top_ids(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25" / "index.pkl"
    indexer = BM25Indexer(index_path)
    indexer.build(corpus())

    before = indexer.query(["apple", "durian"], top_k=2)
    reopened = BM25Indexer(index_path)
    reopened.load()
    after = reopened.query(["apple", "durian"], top_k=2)

    assert index_path.is_file()
    assert [item["id"] for item in before] == ["chunk-a", "chunk-c"]
    assert after == before
    assert not index_path.with_suffix(".tmp").exists()


@pytest.mark.unit
def test_idf_matches_spec_formula_and_posting_shape(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "index.pkl")
    indexer.build(corpus())

    apple = indexer.index["inverted_index"]["apple"]
    expected_idf = math.log((3 - 1 + 0.5) / (1 + 0.5))

    assert apple["idf"] == pytest.approx(expected_idf)
    assert apple["postings"] == [
        {"chunk_id": "chunk-a", "tf": 2.0, "doc_length": 3.0}
    ]


@pytest.mark.unit
def test_build_persists_forward_statistics(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "index.pkl")
    indexer.build(corpus())

    assert indexer.index["format_version"] == 1
    assert indexer.index["document_count"] == 3
    assert indexer.index["total_document_length"] == 8.0
    assert indexer.index["documents"]["chunk-a"] == {
        "text": "text for chunk-a",
        "metadata": {"source_path": "docs/corpus.md"},
        "term_frequencies": {"apple": 2.0, "banana": 1.0},
        "document_length": 3.0,
    }


@pytest.mark.unit
def test_incremental_update_keeps_old_records_and_recalculates_idf(
    tmp_path: Path,
) -> None:
    indexer = BM25Indexer(tmp_path / "index.pkl")
    indexer.build(corpus())

    indexer.update([record("chunk-d", {"eggplant": 1.0})])

    assert indexer.index["document_count"] == 4
    assert set(indexer.index["documents"]) == {
        "chunk-a",
        "chunk-b",
        "chunk-c",
        "chunk-d",
    }
    expected = math.log((4 - 1 + 0.5) / (1 + 0.5))
    assert indexer.index["inverted_index"]["apple"]["idf"] == pytest.approx(
        expected
    )
    assert indexer.query(["eggplant"])[0]["id"] == "chunk-d"


@pytest.mark.unit
def test_incremental_update_replaces_existing_chunk(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "index.pkl")
    indexer.build(corpus())

    indexer.update([record("chunk-a", {"fig": 3.0})])

    assert "apple" not in indexer.index["inverted_index"]
    assert indexer.query(["fig"])[0]["id"] == "chunk-a"


@pytest.mark.unit
def test_rebuild_discards_previous_corpus(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path / "index.pkl")
    indexer.build(corpus())

    indexer.build([record("replacement", {"grape": 1.0})])

    assert set(indexer.index["documents"]) == {"replacement"}
    assert set(indexer.index["inverted_index"]) == {"grape"}


@pytest.mark.unit
def test_remove_document_uses_source_path_and_persists(tmp_path: Path) -> None:
    index_path = tmp_path / "index.pkl"
    indexer = BM25Indexer(index_path)
    indexer.build(
        corpus()
        + [record("other", {"hazelnut": 1.0}, source_path="docs/other.md")]
    )

    removed = indexer.remove_document("docs/corpus.md")
    reopened = BM25Indexer(index_path)

    assert removed == 3
    assert set(reopened.index["documents"]) == {"other"}
    assert reopened.query(["hazelnut"])[0]["id"] == "other"


@pytest.mark.unit
def test_empty_index_roundtrip_and_query(tmp_path: Path) -> None:
    index_path = tmp_path / "index.pkl"
    BM25Indexer(index_path).build([])
    reopened = BM25Indexer(index_path)

    assert reopened.index["document_count"] == 0
    assert reopened.index["inverted_index"] == {}
    assert reopened.query(["anything"]) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "sparse_vector",
    [None, {"": 1.0}, {"term": 0.0}, {"term": -1.0}, {"term": float("nan")}],
)
def test_invalid_sparse_statistics_are_rejected(
    tmp_path: Path, sparse_vector: dict[str, float] | None
) -> None:
    invalid = ChunkRecord(
        id="invalid",
        text="invalid",
        metadata={"source_path": "docs/invalid.md"},
        sparse_vector=sparse_vector,
    )

    with pytest.raises(ValueError, match="sparse"):
        BM25Indexer(tmp_path / "index.pkl").build([invalid])


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [0, -1, True])
def test_query_top_k_must_be_positive(tmp_path: Path, top_k: int) -> None:
    indexer = BM25Indexer(tmp_path / "index.pkl")
    indexer.build(corpus())

    with pytest.raises(ValueError, match="top_k"):
        indexer.query(["apple"], top_k=top_k)
