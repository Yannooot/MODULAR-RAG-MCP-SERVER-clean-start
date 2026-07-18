import json
from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import EmbeddingProviderError, OpenAIEmbedding


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
        embedding=replace(
            settings.embedding,
            provider=provider,
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="https://api.example.test",
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "provider_class", "expected_url", "expected_payload"),
    [
        (
            "openai",
            OpenAIEmbedding,
            "https://api.example.test/embeddings",
            {"model": "text-embedding-3-small", "input": ["first", "second"]},
        ),
        (
            "azure",
            AzureEmbedding,
            "https://api.example.test/openai/deployments/text-embedding-3-small/"
            "embeddings?api-version=2024-02-15-preview",
            {"input": ["first", "second"]},
        ),
    ],
)
def test_builtin_provider_routes_and_embeds_batch(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    provider_class: type[OpenAIEmbedding],
    expected_url: str,
    expected_payload: dict[str, Any],
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse(
            {
                "data": [
                    {"index": 1, "embedding": [2, 3]},
                    {"index": 0, "embedding": [1, 1]},
                ]
            }
        )

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = EmbeddingFactory.create(settings_for(provider))

    vectors = embedding.embed(["first", "second"])

    request = captured["request"]
    assert isinstance(embedding, provider_class)
    assert vectors == [[1.0, 1.0], [2.0, 3.0]]
    assert request.full_url == expected_url
    assert captured["timeout"] == 30
    assert json.loads(request.data) == expected_payload
    if provider == "azure":
        headers = {name.lower(): value for name, value in request.headers.items()}
        assert headers["api-key"] == "test-key"
        assert request.get_header("Authorization") is None
    else:
        assert request.get_header("Authorization") == "Bearer test-key"


@pytest.mark.unit
@pytest.mark.parametrize("texts", [[], [""], ["x" * 100_001]])
def test_invalid_embedding_input_has_readable_error(texts: list[str]) -> None:
    embedding = OpenAIEmbedding(settings_for("openai"))

    with pytest.raises(
        EmbeddingProviderError, match="openai provider InputValidationError"
    ):
        embedding.embed(texts)


@pytest.mark.unit
def test_malformed_embedding_response_has_readable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "libs.embedding.openai_embedding.urlopen",
        lambda request, timeout: FakeHTTPResponse({"data": []}),
    )
    embedding = OpenAIEmbedding(settings_for("openai"))

    with pytest.raises(
        EmbeddingProviderError, match="openai provider ProviderResponseError"
    ):
        embedding.embed(["text"])
