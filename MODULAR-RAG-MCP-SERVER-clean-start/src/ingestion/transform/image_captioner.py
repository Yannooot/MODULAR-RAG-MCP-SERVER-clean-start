"""Generate image captions without blocking ingestion on vision failures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


DEFAULT_PROMPT = (
    "Describe the image accurately. Preserve text, structure, data relationships, "
    "and conclusions useful for retrieval. Return only the description."
)


class ImageCaptioner(BaseTransform):
    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        super().__init__(settings)
        self.enabled = settings.vision_llm.enabled
        self.prompt = self._load_prompt(prompt_path)
        self.vision_llm = vision_llm
        self._setup_error: str | None = None
        if self.enabled and self.vision_llm is None:
            try:
                self.vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception as exc:
                self._setup_error = str(exc)

    def transform(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        results: list[Chunk] = []
        for chunk in chunks:
            refs = self._image_refs(chunk.metadata)
            if not refs:
                results.append(chunk)
                continue
            try:
                results.append(self._caption_chunk(chunk, refs, trace))
            except Exception as exc:
                metadata = dict(chunk.metadata)
                metadata["has_unprocessed_images"] = True
                metadata["image_caption_fallback_reason"] = str(exc)
                results.append(replace(chunk, metadata=metadata))

        if trace is not None:
            trace.record_stage(
                "image_captioner",
                {"chunk_count": len(chunks), "vision_enabled": self.enabled},
            )
        return results

    def _caption_chunk(
        self,
        chunk: Chunk,
        refs: list[str],
        trace: TraceContext | None,
    ) -> Chunk:
        metadata = dict(chunk.metadata)
        captions = self._existing_captions(metadata)
        pending_refs = [ref for ref in refs if ref not in captions]

        if not self.enabled or self.vision_llm is None:
            metadata["has_unprocessed_images"] = bool(pending_refs)
            if pending_refs:
                metadata["unprocessed_image_refs"] = pending_refs
            else:
                metadata.pop("unprocessed_image_refs", None)
                metadata.pop("image_caption_errors", None)
                metadata.pop("image_caption_fallback_reason", None)
            if self._setup_error:
                metadata["image_caption_fallback_reason"] = self._setup_error
            return replace(chunk, metadata=metadata)

        images = {
            item.get("id"): item
            for item in metadata.get("images", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        errors: dict[str, str] = {}
        unprocessed: list[str] = []
        for image_id in pending_refs:
            image = images.get(image_id)
            path = image.get("path") if image else None
            if not isinstance(path, str) or not path.strip():
                unprocessed.append(image_id)
                errors[image_id] = "image record or path is missing"
                continue
            try:
                caption = self.vision_llm.chat_with_image(
                    self.prompt, path, trace
                ).strip()
                if not caption:
                    raise ValueError("empty caption response")
                captions[image_id] = caption
            except Exception as exc:
                unprocessed.append(image_id)
                errors[image_id] = str(exc)

        if captions:
            metadata["image_captions"] = captions
        metadata["has_unprocessed_images"] = bool(unprocessed)
        if unprocessed:
            metadata["unprocessed_image_refs"] = unprocessed
            metadata["image_caption_errors"] = errors
        else:
            metadata.pop("unprocessed_image_refs", None)
            metadata.pop("image_caption_errors", None)
            metadata.pop("image_caption_fallback_reason", None)
        return replace(chunk, metadata=metadata)

    @staticmethod
    def _image_refs(metadata: dict[str, Any]) -> list[str]:
        refs = metadata.get("image_refs")
        if not isinstance(refs, list):
            return []
        valid_refs = [ref for ref in refs if isinstance(ref, str) and ref.strip()]
        return list(dict.fromkeys(valid_refs))

    @staticmethod
    def _existing_captions(metadata: dict[str, Any]) -> dict[str, str]:
        captions = metadata.get("image_captions")
        if not isinstance(captions, dict):
            return {}
        return {
            image_id: caption
            for image_id, caption in captions.items()
            if isinstance(image_id, str)
            and isinstance(caption, str)
            and caption.strip()
        }

    @staticmethod
    def _load_prompt(prompt_path: str | Path | None) -> str:
        path = (
            Path(prompt_path)
            if prompt_path is not None
            else Path(__file__).parents[3] / "config" / "prompts" / "image_captioning.txt"
        )
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError:
            return DEFAULT_PROMPT
        return prompt or DEFAULT_PROMPT
