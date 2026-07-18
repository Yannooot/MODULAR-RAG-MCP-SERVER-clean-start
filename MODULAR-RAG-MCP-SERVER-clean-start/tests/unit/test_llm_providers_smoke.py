import json
from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import LLMProviderError, OpenAILLM


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def settings_for(provider: str) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        llm=replace(
            settings.llm,
            provider=provider,
            model="test-model",
            api_key="test-key",
            base_url="https://api.example.test",
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "provider_class", "expected_url"),
    [
        ("openai", OpenAILLM, "https://api.example.test/chat/completions"),
        ("deepseek", DeepSeekLLM, "https://api.example.test/chat/completions"),
        (
            "azure",
            AzureLLM,
            "https://api.example.test/openai/deployments/test-model/"
            "chat/completions?api-version=2024-02-15-preview",
        ),
    ],
)
def test_builtin_provider_routes_and_sends_chat_request(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    provider_class: type[OpenAILLM],
    expected_url: str,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse({"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = LLMFactory.create(settings_for(provider))

    response = llm.chat([{"role": "user", "content": "hello"}])

    request = captured["request"]
    assert isinstance(llm, provider_class)
    assert response == "answer"
    assert request.full_url == expected_url
    assert captured["timeout"] == 30
    assert json.loads(request.data) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if provider == "azure":
        headers = {name.lower(): value for name, value in request.headers.items()}
        assert headers["api-key"] == "test-key"
        assert request.get_header("Authorization") is None
    else:
        assert request.get_header("Authorization") == "Bearer test-key"


@pytest.mark.unit
def test_chat_rejects_invalid_message_shape_with_provider_and_error_type() -> None:
    llm = DeepSeekLLM(settings_for("deepseek"))

    with pytest.raises(
        LLMProviderError, match="deepseek provider InputValidationError"
    ):
        llm.chat("not a message sequence")


@pytest.mark.unit
def test_chat_rejects_malformed_provider_response_with_readable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "libs.llm.openai_llm.urlopen",
        lambda request, timeout: FakeHTTPResponse({"choices": []}),
    )
    llm = LLMFactory.create(settings_for("openai"))

    with pytest.raises(
        LLMProviderError, match="openai provider ProviderResponseError"
    ):
        llm.chat([{"role": "user", "content": "hello"}])
