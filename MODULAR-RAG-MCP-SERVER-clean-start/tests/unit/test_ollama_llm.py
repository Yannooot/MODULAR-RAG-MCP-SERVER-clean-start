import json
from dataclasses import replace
from typing import Any
from urllib.error import URLError

import pytest

from core.settings import Settings, load_settings
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import LLMProviderError


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def settings_for() -> Settings:
    settings = load_settings()
    return replace(
        settings,
        llm=replace(
            settings.llm,
            provider="ollama",
            model="llama3",
            api_key="not-for-error-output",
            base_url="http://localhost:11434/",
        ),
    )


@pytest.mark.unit
def test_factory_creates_ollama_and_sends_non_streaming_chat_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse({"message": {"content": "answer"}})

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = LLMFactory.create(settings_for())

    response = llm.chat([{"role": "user", "content": "hello"}])

    request = captured["request"]
    assert isinstance(llm, OllamaLLM)
    assert response == "answer"
    assert request.full_url == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 30
    assert request.get_header("Authorization") is None
    assert json.loads(request.data) == {
        "model": "llama3",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


@pytest.mark.unit
@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError("timed out")])
def test_network_errors_are_readable_and_do_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = OllamaLLM(settings_for())

    with pytest.raises(LLMProviderError) as exc_info:
        llm.chat([{"role": "user", "content": "hello"}])

    assert "ollama provider" in str(exc_info.value)
    assert type(error).__name__ in str(exc_info.value)
    assert "not-for-error-output" not in str(exc_info.value)


@pytest.mark.unit
def test_malformed_response_has_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "libs.llm.ollama_llm.urlopen",
        lambda request, timeout: FakeHTTPResponse({"message": {}}),
    )
    llm = OllamaLLM(settings_for())

    with pytest.raises(
        LLMProviderError, match="ollama provider ProviderResponseError"
    ):
        llm.chat([{"role": "user", "content": "hello"}])
