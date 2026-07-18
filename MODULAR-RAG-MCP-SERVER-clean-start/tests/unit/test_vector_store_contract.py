from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
    VectorStoreFactoryError,
)


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.records: list[dict[str, Any]] = []

    def upsert(
        self, records: list[dict[str, Any]], trace: Any | None = None
    ) -> None:
        self.records = list(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        results = self.records
        if filters:
            results = [
                record
                for record in results
                if all(record["metadata"].get(key) == value for key, value in filters.items())
            ]
        return [
            {
                "id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
                "score": 1.0,
            }
            for record in results[:top_k]
        ]


@pytest.fixture(autouse=True)
def isolate_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VectorStoreFactory, "_providers", {})


def settings_for(backend: str) -> Settings:
    settings = load_settings()
    return replace(settings, vector_store=replace(settings.vector_store, backend=backend))


@pytest.mark.unit
def test_upsert_and_query_follow_record_contract() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)
    store = VectorStoreFactory.create(settings_for("fake"))
    store.upsert(
        [
            {"id": "one", "text": "RAG", "metadata": {"topic": "rag"}},
            {"id": "two", "text": "MCP", "metadata": {"topic": "mcp"}},
        ]
    )

    results = store.query([0.1, 0.2], top_k=1, filters={"topic": "mcp"})

    assert results == [
        {"id": "two", "text": "MCP", "metadata": {"topic": "mcp"}, "score": 1.0}
    ]


@pytest.mark.unit
def test_factory_passes_complete_settings_to_provider() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)
    settings = settings_for("fake")

    store = VectorStoreFactory.create(settings)

    assert isinstance(store, FakeVectorStore)
    assert store.settings is settings


@pytest.mark.unit
def test_unknown_provider_has_readable_error() -> None:
    with pytest.raises(VectorStoreFactoryError, match="unknown-provider"):
        VectorStoreFactory.create(settings_for("unknown-provider"))


@pytest.mark.unit
def test_registered_provider_must_implement_base_vector_store() -> None:
    with pytest.raises(TypeError, match="BaseVectorStore"):
        VectorStoreFactory.register_provider("invalid", object)


@pytest.mark.unit
def test_base_vector_store_requires_contract_methods() -> None:
    class IncompleteVectorStore(BaseVectorStore):
        pass

    with pytest.raises(TypeError):
        IncompleteVectorStore(settings_for("incomplete"))
