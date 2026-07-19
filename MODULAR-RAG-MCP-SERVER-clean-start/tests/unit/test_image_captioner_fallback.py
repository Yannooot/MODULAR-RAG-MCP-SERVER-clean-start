from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.llm_factory import LLMFactory


class FakeVisionLLM:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls: list[tuple[str, str | bytes, Any]] = []

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        self.calls.append((text, image_path, trace))
        key = str(image_path)
        if key in self.errors:
            raise self.errors[key]
        return self.responses.get(key, "")


def settings(enabled: bool) -> Settings:
    app_settings = load_settings()
    return replace(
        app_settings,
        vision_llm=replace(app_settings.vision_llm, enabled=enabled),
    )


def image(image_id: str, path: str) -> dict[str, Any]:
    return {
        "id": image_id,
        "path": path,
        "page": 1,
        "text_offset": 0,
        "text_length": 20,
        "position": {},
    }


def chunk_with_images(
    refs: list[str] | None = None,
    images: list[dict[str, Any]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Chunk:
    metadata: dict[str, Any] = {"source_path": "docs/manual.pdf"}
    if refs is not None:
        metadata["image_refs"] = refs
    if images is not None:
        metadata["images"] = images
    if extra_metadata:
        metadata.update(extra_metadata)
    return Chunk(
        id="chunk-1",
        text="系统架构如下图所示。",
        metadata=metadata,
        start_offset=0,
        end_offset=10,
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_enabled_mode_generates_captions_for_each_image() -> None:
    vision = FakeVisionLLM(
        {"images/a.png": "三层系统架构图", "images/b.png": "数据处理流程图"}
    )
    source = chunk_with_images(
        ["image-a", "image-b"],
        [image("image-a", "images/a.png"), image("image-b", "images/b.png")],
    )

    result = ImageCaptioner(settings(True), vision_llm=vision).transform([source])[0]

    assert result.metadata["image_captions"] == {
        "image-a": "三层系统架构图",
        "image-b": "数据处理流程图",
    }
    assert result.metadata["has_unprocessed_images"] is False
    assert [call[1] for call in vision.calls] == ["images/a.png", "images/b.png"]
    assert all(call[0] for call in vision.calls)
    assert source.metadata.get("image_captions") is None


@pytest.mark.unit
def test_disabled_mode_keeps_refs_and_marks_images_unprocessed() -> None:
    vision = FakeVisionLLM(errors={"images/a.png": AssertionError("not called")})
    source = chunk_with_images(
        ["image-a"], [image("image-a", "images/a.png")]
    )

    result = ImageCaptioner(settings(False), vision_llm=vision).transform([source])[0]

    assert result.metadata["image_refs"] == ["image-a"]
    assert result.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in result.metadata
    assert vision.calls == []


@pytest.mark.unit
def test_chunk_without_image_refs_is_unchanged() -> None:
    source = chunk_with_images()

    result = ImageCaptioner(settings(False)).transform([source])[0]

    assert result == source
    assert "has_unprocessed_images" not in result.metadata


@pytest.mark.unit
def test_one_image_failure_keeps_successful_caption() -> None:
    vision = FakeVisionLLM(
        responses={"images/a.png": "架构图"},
        errors={"images/b.png": RuntimeError("vision offline")},
    )
    source = chunk_with_images(
        ["image-a", "image-b"],
        [image("image-a", "images/a.png"), image("image-b", "images/b.png")],
    )

    result = ImageCaptioner(settings(True), vision_llm=vision).transform([source])[0]

    assert result.metadata["image_captions"] == {"image-a": "架构图"}
    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["unprocessed_image_refs"] == ["image-b"]
    assert "vision offline" in result.metadata["image_caption_errors"]["image-b"]


@pytest.mark.unit
def test_missing_image_record_is_marked_unprocessed() -> None:
    result = ImageCaptioner(
        settings(True), vision_llm=FakeVisionLLM()
    ).transform([chunk_with_images(["missing"], [])])[0]

    assert result.metadata["image_refs"] == ["missing"]
    assert result.metadata["unprocessed_image_refs"] == ["missing"]
    assert "missing" in result.metadata["image_caption_errors"]["missing"]


@pytest.mark.unit
def test_empty_caption_is_treated_as_failure() -> None:
    result = ImageCaptioner(
        settings(True), vision_llm=FakeVisionLLM({"images/a.png": "  "})
    ).transform(
        [chunk_with_images(["image-a"], [image("image-a", "images/a.png")])]
    )[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert "empty" in result.metadata["image_caption_errors"]["image-a"]


@pytest.mark.unit
def test_existing_caption_is_not_generated_again() -> None:
    vision = FakeVisionLLM(errors={"images/a.png": AssertionError("not called")})
    source = chunk_with_images(
        ["image-a"],
        [image("image-a", "images/a.png")],
        {"image_captions": {"image-a": "已有描述"}},
    )

    first = ImageCaptioner(settings(True), vision_llm=vision).transform([source])[0]
    second = ImageCaptioner(settings(True), vision_llm=vision).transform([first])[0]

    assert first.metadata["image_captions"] == {"image-a": "已有描述"}
    assert second.metadata == first.metadata
    assert vision.calls == []


@pytest.mark.unit
def test_duplicate_image_ref_is_generated_once() -> None:
    vision = FakeVisionLLM({"images/a.png": "架构图"})

    result = ImageCaptioner(settings(True), vision_llm=vision).transform(
        [
            chunk_with_images(
                ["image-a", "image-a"],
                [image("image-a", "images/a.png")],
            )
        ]
    )[0]

    assert result.metadata["image_refs"] == ["image-a", "image-a"]
    assert result.metadata["image_captions"] == {"image-a": "架构图"}
    assert len(vision.calls) == 1


@pytest.mark.unit
def test_injected_prompt_file_is_used(tmp_path: Path) -> None:
    prompt = tmp_path / "caption.txt"
    prompt.write_text("请描述图片中的检索信息。", encoding="utf-8")
    vision = FakeVisionLLM({"images/a.png": "图片描述"})

    ImageCaptioner(
        settings(True), vision_llm=vision, prompt_path=prompt
    ).transform(
        [chunk_with_images(["image-a"], [image("image-a", "images/a.png")])]
    )

    assert vision.calls[0][0] == "请描述图片中的检索信息。"


@pytest.mark.unit
def test_factory_failure_degrades_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LLMFactory,
        "create_vision_llm",
        lambda app_settings: (_ for _ in ()).throw(ValueError("unknown provider")),
    )

    result = ImageCaptioner(settings(True)).transform(
        [chunk_with_images(["image-a"], [image("image-a", "images/a.png")])]
    )[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert "unknown provider" in result.metadata["image_caption_fallback_reason"]


@pytest.mark.unit
def test_transform_records_trace_summary() -> None:
    trace = TraceContext()

    ImageCaptioner(settings(False)).transform(
        [chunk_with_images(["image-a"], [image("image-a", "images/a.png")])],
        trace,
    )

    assert trace.stages[-1] == {
        "name": "image_captioner",
        "details": {"chunk_count": 1, "vision_enabled": False},
    }
