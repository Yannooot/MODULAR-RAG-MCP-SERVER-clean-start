"""Ollama local chat provider."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.openai_llm import LLMProviderError


class OllamaLLM(BaseLLM):
    provider_name = "ollama"
    timeout_seconds = 30

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        normalized_messages = self._validate_messages(messages)
        model, base_url = self._validate_configuration()
        request = Request(
            f"{base_url}/api/chat",
            data=json.dumps(
                {"model": model, "messages": normalized_messages, "stream": False}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._extract_content(self._send_request(request))

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

    def _validate_configuration(self) -> tuple[str, str]:
        model = self.settings.llm.model
        base_url = self.settings.llm.base_url
        if not isinstance(model, str) or not model.strip():
            self._raise("ConfigurationError", "model must be configured")
        if not isinstance(base_url, str) or not base_url.strip():
            self._raise("ConfigurationError", "base_url must be configured")
        return model.strip(), base_url.rstrip("/")

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
            content = payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            self._raise("ProviderResponseError", str(exc))
        if not isinstance(content, str):
            self._raise("ProviderResponseError", "response content must be a string")
        return content

    def _raise(self, error_type: str, detail: str) -> None:
        raise LLMProviderError(f"{self.provider_name} provider {error_type}: {detail}")
