from dataclasses import replace

import pytest

from core.settings import Settings, load_settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory, LLMFactoryError


class FakeLLM(BaseLLM):
    def chat(self, messages: list[dict[str, str]]) -> str:
        return messages[-1]["content"]


@pytest.fixture(autouse=True)
def isolate_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLMFactory, "_providers", {})


def settings_for(provider: str) -> Settings:
    settings = load_settings()
    return replace(settings, llm=replace(settings.llm, provider=provider))


@pytest.mark.unit
def test_factory_routes_to_registered_provider() -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    settings = settings_for("fake")

    llm = LLMFactory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.settings is settings
    assert llm.chat([{"role": "user", "content": "hello"}]) == "hello"


@pytest.mark.unit
def test_provider_names_are_case_insensitive() -> None:
    LLMFactory.register_provider("MixedCase", FakeLLM)

    assert isinstance(LLMFactory.create(settings_for("mixedcase")), FakeLLM)


@pytest.mark.unit
def test_unknown_provider_has_readable_error() -> None:
    with pytest.raises(LLMFactoryError, match="unknown-provider"):
        LLMFactory.create(settings_for("unknown-provider"))


@pytest.mark.unit
def test_registered_provider_must_implement_base_llm() -> None:
    with pytest.raises(TypeError, match="BaseLLM"):
        LLMFactory.register_provider("invalid", object)


@pytest.mark.unit
def test_base_llm_requires_chat_implementation() -> None:
    class IncompleteLLM(BaseLLM):
        pass

    with pytest.raises(TypeError):
        IncompleteLLM(settings_for("incomplete"))
