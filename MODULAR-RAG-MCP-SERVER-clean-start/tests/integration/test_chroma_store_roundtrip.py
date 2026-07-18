from dataclasses import replace
from pathlib import Path

import pytest

from core.settings import Settings, load_settings
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


def settings_for(persist_path: Path) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        vector_store=replace(
            settings.vector_store,
            backend="chroma",
            persist_path=str(persist_path),
            collection_name="roundtrip",
        ),
    )


def sample_records() -> list[dict[str, object]]:
    return [
        {
            "id": "rag",
            "text": "Retrieval augmented generation",
            "metadata": {"topic": "rag", "order": 1},
            "vector": [1.0, 0.0],
        },
        {
            "id": "hybrid",
            "text": "Hybrid retrieval",
            "metadata": {"topic": "rag", "order": 2},
            "dense_vector": [0.8, 0.2],
        },
        {
            "id": "mcp",
            "text": "Model Context Protocol",
            "metadata": {"topic": "mcp", "order": 3},
            "vector": [0.0, 1.0],
        },
    ]


@pytest.mark.integration
def test_factory_creates_chroma_and_roundtrips_records(tmp_path: Path) -> None:
    settings = settings_for(tmp_path / "chroma")
    store = VectorStoreFactory.create(settings)

    store.upsert(sample_records())
    results = store.query([1.0, 0.0], top_k=2)

    assert isinstance(store, ChromaStore)
    assert [result["id"] for result in results] == ["rag", "hybrid"]
    assert results[0] == {
        "id": "rag",
        "text": "Retrieval augmented generation",
        "metadata": {"topic": "rag", "order": 1},
        "score": pytest.approx(1.0),
    }


@pytest.mark.integration
def test_query_honors_top_k_and_metadata_filters(tmp_path: Path) -> None:
    store = ChromaStore(settings_for(tmp_path / "chroma"))
    store.upsert(sample_records())

    limited = store.query([1.0, 0.0], top_k=1)
    filtered = store.query([1.0, 0.0], top_k=3, filters={"topic": "mcp"})

    assert [result["id"] for result in limited] == ["rag"]
    assert [result["id"] for result in filtered] == ["mcp"]


@pytest.mark.integration
def test_records_persist_and_existing_ids_are_updated(tmp_path: Path) -> None:
    settings = settings_for(tmp_path / "chroma")
    store = ChromaStore(settings)
    store.upsert(sample_records())
    store.upsert(
        [
            {
                "id": "rag",
                "text": "Updated RAG text",
                "metadata": {"topic": "updated"},
                "vector": [1.0, 0.0],
            }
        ]
    )

    reopened_store = ChromaStore(settings)
    results = reopened_store.query(
        [1.0, 0.0], top_k=3, filters={"topic": "updated"}
    )

    assert len(results) == 1
    assert results[0]["id"] == "rag"
    assert results[0]["text"] == "Updated RAG text"
