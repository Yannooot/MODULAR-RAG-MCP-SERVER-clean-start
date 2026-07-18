import base64
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from core.settings import Settings, load_settings
from libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from libs.llm.llm_factory import LLMFactory
from PIL import Image


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def settings_for(max_image_size: int = 2048) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        vision_llm=replace(
            settings.vision_llm,
            provider="azure",
            model="gpt-4o",
            api_key="test-key",
            azure_endpoint="https://vision.example.test",
            api_version="2024-02-15-preview",
            deployment_name="vision deployment",
            max_image_size=max_image_size,
            max_tokens=500,
        ),
    )


def write_png(path: Path, size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color="white").save(buffer, format="PNG")
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


@pytest.mark.unit
def test_factory_creates_builtin_azure_vision_provider() -> None:
    assert isinstance(LLMFactory.create_vision_llm(settings_for()), AzureVisionLLM)


@pytest.mark.unit
def test_path_input_sends_azure_multimodal_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "chart.png"
    write_png(image_path, (10, 8))
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse({"choices": [{"message": {"content": "a chart"}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)

    response = AzureVisionLLM(settings_for()).chat_with_image(
        "Describe the image", str(image_path)
    )

    request = captured["request"]
    payload = json.loads(request.data)
    headers = {name.lower(): value for name, value in request.headers.items()}
    assert response == "a chart"
    assert request.full_url == (
        "https://vision.example.test/openai/deployments/vision%20deployment/"
        "chat/completions?api-version=2024-02-15-preview"
    )
    assert headers["api-key"] == "test-key"
    assert captured["timeout"] == 30
    assert payload["model"] == "vision deployment"
    assert payload["max_tokens"] == 500
    assert payload["messages"][0]["content"][0] == {
        "type": "text",
        "text": "Describe the image",
    }
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


@pytest.mark.unit
def test_base64_input_is_compressed_to_max_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_image = write_png(tmp_path / "large.png", (20, 10))
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["payload"] = json.loads(request.data)
        return FakeHTTPResponse({"choices": [{"message": {"content": "done"}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)

    AzureVisionLLM(settings_for(max_image_size=8)).chat_with_image(
        "Describe", base64.b64encode(raw_image)
    )

    image_url = captured["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    encoded = image_url.split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (8, 4)


@pytest.mark.unit
def test_timeout_has_readable_azure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "libs.llm.azure_vision_llm.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("timed out")),
    )

    with pytest.raises(AzureVisionLLMError, match="timed out"):
        AzureVisionLLM(settings_for()).chat_with_image(
            "Describe", base64.b64encode(write_image_bytes())
        )


@pytest.mark.unit
def test_authentication_error_includes_azure_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_body = BytesIO(
        json.dumps(
            {"error": {"code": "Unauthorized", "message": "Invalid API key"}}
        ).encode("utf-8")
    )
    error = HTTPError(
        "https://vision.example.test", 401, "Unauthorized", {}, error_body
    )
    monkeypatch.setattr(
        "libs.llm.azure_vision_llm.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(error),
    )

    with pytest.raises(AzureVisionLLMError, match="Unauthorized.*Invalid API key"):
        AzureVisionLLM(settings_for()).chat_with_image(
            "Describe", base64.b64encode(write_image_bytes())
        )


def write_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="white").save(buffer, format="PNG")
    return buffer.getvalue()
