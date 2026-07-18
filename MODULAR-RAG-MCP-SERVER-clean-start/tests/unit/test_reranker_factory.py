from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.reranker_factory import RerankerFactory, RerankerFactoryError


class FakeReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        return list(reversed(candidates))


@pytest.fixture(autouse=True)
def isolate_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RerankerFactory, "_providers", {"none": NoneReranker})


def settings_for(backend: str) -> Settings:
    settings = load_settings()
    return replace(settings, rerank=replace(settings.rerank, backend=backend))


@pytest.mark.unit
def test_none_backend_keeps_candidate_order() -> None:
    candidates = [{"id": "first"}, {"id": "second"}]

    reranker = RerankerFactory.create(settings_for("none"))
    ranked = reranker.rerank("query", candidates)

    assert isinstance(reranker, NoneReranker)
    assert ranked is candidates
    assert [candidate["id"] for candidate in ranked] == ["first", "second"]


@pytest.mark.unit
def test_factory_routes_to_registered_backend() -> None:
    RerankerFactory.register_provider("fake", FakeReranker)
    settings = settings_for("fake")

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, FakeReranker)
    assert reranker.settings is settings
    assert reranker.rerank("query", [{"id": "first"}, {"id": "second"}]) == [
        {"id": "second"},
        {"id": "first"},
    ]


@pytest.mark.unit
def test_unknown_backend_has_readable_error() -> None:
    with pytest.raises(RerankerFactoryError, match="unknown-backend"):
        RerankerFactory.create(settings_for("unknown-backend"))


@pytest.mark.unit
def test_registered_provider_must_implement_base_reranker() -> None:
    with pytest.raises(TypeError, match="BaseReranker"):
        RerankerFactory.register_provider("invalid", object)


@pytest.mark.unit
def test_base_reranker_requires_rerank_implementation() -> None:
    class IncompleteReranker(BaseReranker):
        pass

    with pytest.raises(TypeError):
        IncompleteReranker(settings_for("none"))
