import json
from dataclasses import replace
from typing import Any
from urllib.error import URLError

import pytest

from core.settings import Settings, load_settings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.openai_embedding import EmbeddingProviderError


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
        embedding=replace(
            settings.embedding,
            provider="ollama",
            model="nomic-embed-text",
            api_key="not-for-error-output",
            base_url="http://localhost:11434/",
        ),
    )


@pytest.mark.unit
def test_factory_creates_ollama_and_embeds_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse({"embeddings": [[1, 2], [3, 4]]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = EmbeddingFactory.create(settings_for())

    vectors = embedding.embed(["first", "second"])

    request = captured["request"]
    assert isinstance(embedding, OllamaEmbedding)
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert request.full_url == "http://localhost:11434/api/embed"
    assert captured["timeout"] == 30
    assert request.get_header("Authorization") is None
    assert json.loads(request.data) == {
        "model": "nomic-embed-text",
        "input": ["first", "second"],
        "truncate": False,
    }


@pytest.mark.unit
@pytest.mark.parametrize("texts", [[], ["x" * 100_001]])
def test_invalid_input_has_readable_error(texts: list[str]) -> None:
    embedding = OllamaEmbedding(settings_for())

    with pytest.raises(
        EmbeddingProviderError, match="ollama provider InputValidationError"
    ):
        embedding.embed(texts)


@pytest.mark.unit
@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError("timed out")])
def test_network_errors_are_readable_and_do_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding(settings_for())

    with pytest.raises(EmbeddingProviderError) as exc_info:
        embedding.embed(["text"])

    assert "ollama provider" in str(exc_info.value)
    assert type(error).__name__ in str(exc_info.value)
    assert "not-for-error-output" not in str(exc_info.value)


@pytest.mark.unit
def test_response_vector_count_must_match_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "libs.embedding.openai_embedding.urlopen",
        lambda request, timeout: FakeHTTPResponse({"embeddings": [[1, 2]]}),
    )
    embedding = OllamaEmbedding(settings_for())

    with pytest.raises(
        EmbeddingProviderError, match="ollama provider ProviderResponseError"
    ):
        embedding.embed(["first", "second"])
