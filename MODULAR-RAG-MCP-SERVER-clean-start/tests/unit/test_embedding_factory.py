from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError


class FakeEmbedding(BaseEmbedding):
    def embed(
        self, texts: list[str], trace: Any | None = None
    ) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


@pytest.fixture(autouse=True)
def isolate_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EmbeddingFactory, "_providers", {})


def settings_for(provider: str) -> Settings:
    settings = load_settings()
    return replace(settings, embedding=replace(settings.embedding, provider=provider))


@pytest.mark.unit
def test_factory_routes_to_registered_batch_provider() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)
    settings = settings_for("fake")

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings is settings
    assert embedding.embed(["a", "abcd"]) == [[1.0, 1.0], [4.0, 1.0]]


@pytest.mark.unit
def test_provider_names_are_case_insensitive() -> None:
    EmbeddingFactory.register_provider("MixedCase", FakeEmbedding)

    assert isinstance(
        EmbeddingFactory.create(settings_for("mixedcase")), FakeEmbedding
    )


@pytest.mark.unit
def test_unknown_provider_has_readable_error() -> None:
    with pytest.raises(EmbeddingFactoryError, match="unknown-provider"):
        EmbeddingFactory.create(settings_for("unknown-provider"))


@pytest.mark.unit
def test_registered_provider_must_implement_base_embedding() -> None:
    with pytest.raises(TypeError, match="BaseEmbedding"):
        EmbeddingFactory.register_provider("invalid", object)


@pytest.mark.unit
def test_base_embedding_requires_embed_implementation() -> None:
    class IncompleteEmbedding(BaseEmbedding):
        pass

    with pytest.raises(TypeError):
        IncompleteEmbedding(settings_for("incomplete"))
