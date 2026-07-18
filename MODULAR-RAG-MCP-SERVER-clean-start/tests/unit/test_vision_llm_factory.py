from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory, LLMFactoryError


class FakeVisionLLM(BaseVisionLLM):
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        image = self.preprocess_image(image_path)
        return f"{text}:{image!r}"


@pytest.fixture(autouse=True)
def isolate_vision_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLMFactory, "_vision_providers", {})


def settings_for(provider: str) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        vision_llm=replace(settings.vision_llm, provider=provider),
    )


@pytest.mark.unit
@pytest.mark.parametrize("image", ["images/chart.png", b"base64-image"])
def test_factory_routes_path_and_bytes_to_registered_provider(
    image: str | bytes,
) -> None:
    LLMFactory.register_vision_provider("fake", FakeVisionLLM)
    settings = settings_for("fake")

    vision_llm = LLMFactory.create_vision_llm(settings)

    assert isinstance(vision_llm, FakeVisionLLM)
    assert vision_llm.settings is settings
    assert vision_llm.chat_with_image("describe", image) == f"describe:{image!r}"


@pytest.mark.unit
def test_vision_provider_names_are_case_insensitive() -> None:
    LLMFactory.register_vision_provider("MixedCase", FakeVisionLLM)

    assert isinstance(
        LLMFactory.create_vision_llm(settings_for("mixedcase")), FakeVisionLLM
    )


@pytest.mark.unit
def test_unknown_vision_provider_has_readable_error() -> None:
    with pytest.raises(LLMFactoryError, match="unknown-provider"):
        LLMFactory.create_vision_llm(settings_for("unknown-provider"))


@pytest.mark.unit
def test_registered_vision_provider_must_implement_base_class() -> None:
    with pytest.raises(TypeError, match="BaseVisionLLM"):
        LLMFactory.register_vision_provider("invalid", object)


@pytest.mark.unit
def test_default_image_preprocessor_keeps_input_unchanged() -> None:
    vision_llm = FakeVisionLLM(settings_for("fake"))

    assert vision_llm.preprocess_image("image.png") == "image.png"
    assert vision_llm.preprocess_image(b"image-bytes") == b"image-bytes"


@pytest.mark.unit
def test_base_vision_llm_requires_chat_with_image_implementation() -> None:
    class IncompleteVisionLLM(BaseVisionLLM):
        pass

    with pytest.raises(TypeError):
        IncompleteVisionLLM(settings_for("incomplete"))
