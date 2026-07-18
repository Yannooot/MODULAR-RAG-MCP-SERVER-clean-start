"""Ollama local embedding provider."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.request import Request

from libs.embedding.openai_embedding import OpenAIEmbedding

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class OllamaEmbedding(OpenAIEmbedding):
    provider_name = "ollama"

    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        normalized_texts = self._validate_texts(texts)
        model, base_url = self._validate_ollama_configuration()
        request = Request(
            f"{base_url}/api/embed",
            data=json.dumps(
                {"model": model, "input": normalized_texts, "truncate": False}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._extract_ollama_vectors(
            self._send_request(request), len(normalized_texts)
        )

    def _validate_ollama_configuration(self) -> tuple[str, str]:
        model = self.settings.embedding.model
        base_url = self.settings.embedding.base_url
        if not isinstance(model, str) or not model.strip():
            self._raise("ConfigurationError", "model must be configured")
        if not isinstance(base_url, str) or not base_url.strip():
            self._raise("ConfigurationError", "base_url must be configured")
        return model.strip(), base_url.rstrip("/")

    def _extract_ollama_vectors(
        self, payload: Any, text_count: int
    ) -> list[list[float]]:
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != text_count:
            self._raise(
                "ProviderResponseError", "response embedding count does not match input"
            )

        vectors: list[list[float]] = []
        vector_dimension: int | None = None
        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                self._raise(
                    "ProviderResponseError", "response embedding must be a non-empty list"
                )
            if any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in embedding
            ):
                self._raise(
                    "ProviderResponseError", "response embedding values must be numeric"
                )
            if vector_dimension is None:
                vector_dimension = len(embedding)
            elif len(embedding) != vector_dimension:
                self._raise(
                    "ProviderResponseError", "response embedding dimensions must match"
                )
            vectors.append([float(value) for value in embedding])
        return vectors
