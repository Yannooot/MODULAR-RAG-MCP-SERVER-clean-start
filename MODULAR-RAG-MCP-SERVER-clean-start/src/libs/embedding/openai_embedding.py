"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider request cannot be completed."""


class OpenAIEmbedding(BaseEmbedding):
    provider_name = "openai"
    timeout_seconds = 30
    max_text_length = 100_000

    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        normalized_texts = self._validate_texts(texts)
        api_key, model, base_url = self._validate_configuration()
        request = Request(
            self._endpoint(base_url, model),
            data=json.dumps(self._payload(model, normalized_texts)).encode("utf-8"),
            headers=self._headers(api_key),
            method="POST",
        )
        return self._extract_vectors(self._send_request(request), len(normalized_texts))

    def _validate_texts(self, texts: list[str]) -> list[str]:
        if not isinstance(texts, list) or not texts:
            self._raise("InputValidationError", "texts must be a non-empty list")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                self._raise(
                    "InputValidationError", f"text {index} must be a non-empty string"
                )
            if len(text) > self.max_text_length:
                self._raise(
                    "InputValidationError",
                    f"text {index} exceeds {self.max_text_length} characters",
                )
        return texts

    def _validate_configuration(self) -> tuple[str, str, str]:
        api_key = self.settings.embedding.api_key
        model = self.settings.embedding.model
        base_url = self.settings.embedding.base_url
        if not isinstance(api_key, str) or not api_key.strip():
            self._raise("ConfigurationError", "api_key must be configured")
        if not isinstance(model, str) or not model.strip():
            self._raise("ConfigurationError", "model must be configured")
        if not isinstance(base_url, str) or not base_url.strip():
            self._raise("ConfigurationError", "base_url must be configured")
        return api_key.strip(), model.strip(), base_url.rstrip("/")

    def _endpoint(self, base_url: str, model: str) -> str:
        return f"{base_url}/embeddings"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, model: str, texts: list[str]) -> dict[str, Any]:
        return {"model": model, "input": texts}

    def _send_request(self, request: Request) -> Any:
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read()
        except (HTTPError, URLError, OSError) as exc:
            self._raise(type(exc).__name__, str(exc))

        try:
            return json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._raise("ResponseDecodeError", str(exc))

    def _extract_vectors(self, payload: Any, text_count: int) -> list[list[float]]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != text_count:
            self._raise("ProviderResponseError", "response data count does not match input")

        vectors: list[list[float] | None] = [None] * text_count
        for item in data:
            if not isinstance(item, dict):
                self._raise("ProviderResponseError", "response data item must be a mapping")
            index = item.get("index")
            vector = item.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= text_count
                or vectors[index] is not None
            ):
                self._raise("ProviderResponseError", "response embedding index is invalid")
            if not isinstance(vector, list) or not vector:
                self._raise("ProviderResponseError", "response embedding must be a non-empty list")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector):
                self._raise("ProviderResponseError", "response embedding values must be numeric")
            vectors[index] = [float(value) for value in vector]

        if any(vector is None for vector in vectors):
            self._raise("ProviderResponseError", "response is missing an embedding")
        return [vector for vector in vectors if vector is not None]

    def _raise(self, error_type: str, detail: str) -> None:
        raise EmbeddingProviderError(
            f"{self.provider_name} provider {error_type}: {detail}"
        )
