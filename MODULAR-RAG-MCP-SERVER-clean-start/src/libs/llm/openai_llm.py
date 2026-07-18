"""OpenAI-compatible chat provider."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import Settings
from libs.llm.base_llm import BaseLLM


class LLMProviderError(RuntimeError):
    """Raised when a provider request cannot be completed."""


class OpenAILLM(BaseLLM):
    provider_name = "openai"
    timeout_seconds = 30

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        normalized_messages = self._validate_messages(messages)
        api_key, model, base_url = self._validate_configuration()
        request = Request(
            self._endpoint(base_url, model),
            data=json.dumps(self._payload(model, normalized_messages)).encode("utf-8"),
            headers=self._headers(api_key),
            method="POST",
        )
        payload = self._send_request(request)
        return self._extract_content(payload)

    def _validate_messages(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes))
            or not messages
        ):
            self._raise("InputValidationError", "messages must be a non-empty sequence")

        normalized_messages: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                self._raise("InputValidationError", f"message {index} must be a mapping")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                self._raise(
                    "InputValidationError",
                    f"message {index} requires string role and content",
                )
            normalized_messages.append({"role": role, "content": content})
        return normalized_messages

    def _validate_configuration(self) -> tuple[str, str, str]:
        api_key = self.settings.llm.api_key
        model = self.settings.llm.model
        base_url = self.settings.llm.base_url
        if not isinstance(api_key, str) or not api_key.strip():
            self._raise("ConfigurationError", "api_key must be configured")
        if not isinstance(model, str) or not model.strip():
            self._raise("ConfigurationError", "model must be configured")
        if not isinstance(base_url, str) or not base_url.strip():
            self._raise("ConfigurationError", "base_url must be configured")
        return api_key.strip(), model.strip(), base_url.rstrip("/")

    def _endpoint(self, base_url: str, model: str) -> str:
        return f"{base_url}/chat/completions"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self, model: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        return {"model": model, "messages": messages}

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

    def _extract_content(self, payload: Any) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            self._raise("ProviderResponseError", str(exc))
        if not isinstance(content, str):
            self._raise("ProviderResponseError", "response content must be a string")
        return content

    def _raise(self, error_type: str, detail: str) -> None:
        raise LLMProviderError(f"{self.provider_name} provider {error_type}: {detail}")
