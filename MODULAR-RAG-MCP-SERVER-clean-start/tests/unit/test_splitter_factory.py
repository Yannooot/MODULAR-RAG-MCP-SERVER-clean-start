from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory, SplitterFactoryError


class FakeRecursiveSplitter(BaseSplitter):
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        return [part.strip() for part in text.split("|")]


class FakeSemanticSplitter(BaseSplitter):
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        return [text]


@pytest.fixture(autouse=True)
def isolate_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SplitterFactory, "_providers", {})


def settings_for(provider: str) -> Settings:
    settings = load_settings()
    return replace(settings, splitter=replace(settings.splitter, provider=provider))


@pytest.mark.unit
def test_factory_routes_to_different_registered_strategies() -> None:
    SplitterFactory.register_provider("recursive", FakeRecursiveSplitter)
    SplitterFactory.register_provider("semantic", FakeSemanticSplitter)

    recursive = SplitterFactory.create(settings_for("recursive"))
    semantic = SplitterFactory.create(settings_for("semantic"))

    assert isinstance(recursive, FakeRecursiveSplitter)
    assert isinstance(semantic, FakeSemanticSplitter)


@pytest.mark.unit
def test_split_text_returns_stable_chunks() -> None:
    SplitterFactory.register_provider("fake", FakeRecursiveSplitter)
    settings = settings_for("fake")

    splitter = SplitterFactory.create(settings)

    assert splitter.settings is settings
    assert splitter.split_text("first | second") == ["first", "second"]


@pytest.mark.unit
def test_unknown_provider_has_readable_error() -> None:
    with pytest.raises(SplitterFactoryError, match="unknown-provider"):
        SplitterFactory.create(settings_for("unknown-provider"))


@pytest.mark.unit
def test_registered_provider_must_implement_base_splitter() -> None:
    with pytest.raises(TypeError, match="BaseSplitter"):
        SplitterFactory.register_provider("invalid", object)


@pytest.mark.unit
def test_base_splitter_requires_split_text_implementation() -> None:
    class IncompleteSplitter(BaseSplitter):
        pass

    with pytest.raises(TypeError):
        IncompleteSplitter(settings_for("incomplete"))
