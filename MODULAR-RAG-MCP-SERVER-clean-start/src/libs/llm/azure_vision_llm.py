"""Azure OpenAI vision provider."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image

from core.settings import Settings
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse


class AzureVisionLLMError(RuntimeError):
    """Raised when an Azure Vision request cannot be completed."""


class AzureVisionLLM(BaseVisionLLM):
    timeout_seconds = 30

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        endpoint, api_version, deployment, api_key, max_tokens = (
            self._validate_configuration()
        )
        if not isinstance(text, str) or not text.strip():
            raise AzureVisionLLMError("Azure Vision text must be a non-empty string")

        image_url = self.preprocess_image(image_path)
        request = Request(
            self._endpoint(endpoint, deployment, api_version),
            data=json.dumps(
                {
                    "model": deployment,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                }
            ).encode("utf-8"),
            headers={"api-key": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        return self._extract_content(self._send_request(request))

    def preprocess_image(self, image_path: str | bytes) -> str:
        raw_image = self._read_image(image_path)
        try:
            with Image.open(BytesIO(raw_image)) as source:
                source.load()
                image_format = source.format or "PNG"
                mime_type = Image.MIME.get(image_format, "image/png")
                if max(source.size) > self.settings.vision_llm.max_image_size:
                    resized = source.copy()
                    resized.thumbnail(
                        (
                            self.settings.vision_llm.max_image_size,
                            self.settings.vision_llm.max_image_size,
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    if image_format == "JPEG" and resized.mode not in ("RGB", "L"):
                        resized = resized.convert("RGB")
                    output = BytesIO()
                    resized.save(output, format=image_format)
                    raw_image = output.getvalue()
        except (OSError, ValueError) as exc:
            raise AzureVisionLLMError(f"Invalid image input: {exc}") from exc

        encoded = base64.b64encode(raw_image).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _read_image(image_path: str | bytes) -> bytes:
        if isinstance(image_path, str):
            path = Path(image_path)
            if path.is_file():
                return path.read_bytes()
            encoded = image_path.split(",", 1)[1] if image_path.startswith("data:") and "," in image_path else image_path
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise AzureVisionLLMError(
                    f"Image path does not exist or base64 is invalid: {image_path}"
                ) from exc
        if isinstance(image_path, bytes):
            try:
                return base64.b64decode(image_path, validate=True)
            except (ValueError, base64.binascii.Error):
                return image_path
        raise AzureVisionLLMError("Image input must be a path or base64 bytes")

    def _validate_configuration(self) -> tuple[str, str, str, str, int]:
        config = self.settings.vision_llm
        required = {
            "azure_endpoint": config.azure_endpoint,
            "api_version": config.api_version,
            "deployment_name": config.deployment_name,
            "api_key": config.api_key,
        }
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise AzureVisionLLMError(f"vision_llm.{name} must be configured")
        if config.max_image_size <= 0:
            raise AzureVisionLLMError("vision_llm.max_image_size must be positive")
        if config.max_tokens <= 0:
            raise AzureVisionLLMError("vision_llm.max_tokens must be positive")
        return (
            config.azure_endpoint.rstrip("/"),
            config.api_version.strip(),
            config.deployment_name.strip(),
            config.api_key.strip(),
            config.max_tokens,
        )

    @staticmethod
    def _endpoint(endpoint: str, deployment: str, api_version: str) -> str:
        return (
            f"{endpoint}/openai/deployments/{quote(deployment, safe='')}/"
            f"chat/completions?api-version={quote(api_version, safe='')}"
        )

    def _send_request(self, request: Request) -> Any:
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read()
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, TimeoutError, OSError) as exc:
            raise AzureVisionLLMError(f"Azure Vision request failed: {exc}") from exc

        try:
            return json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AzureVisionLLMError(
                f"Azure Vision response decode failed: {exc}"
            ) from exc

    @staticmethod
    def _raise_http_error(exc: HTTPError) -> NoReturn:
        azure_code = f"HTTP {exc.code}"
        detail = str(exc)
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            error = payload.get("error", {})
            azure_code = error.get("code") or azure_code
            detail = error.get("message") or detail
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise AzureVisionLLMError(
            f"Azure Vision {azure_code} (HTTP {exc.code}): {detail}"
        ) from exc

    @staticmethod
    def _extract_content(payload: Any) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise AzureVisionLLMError(
                f"Azure Vision response is missing content: {exc}"
            ) from exc
        if not isinstance(content, str):
            raise AzureVisionLLMError("Azure Vision response content must be a string")
        return content
