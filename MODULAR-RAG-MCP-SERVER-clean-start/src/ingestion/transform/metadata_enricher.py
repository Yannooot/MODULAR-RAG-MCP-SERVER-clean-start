"""Generate searchable metadata for chunks with rule and LLM strategies."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


METADATA_PROMPT = """Analyze the text and return only one JSON object.
Required schema: {{"title": "concise title", "summary": "factual summary", "tags": ["tag1", "tag2"]}}
Use the same language as the text. Include 2 to 5 useful search tags.

Text:
{text}
"""


class MetadataEnricher(BaseTransform):
    def __init__(self, settings: Settings, llm: BaseLLM | None = None) -> None:
        super().__init__(settings)
        self.use_llm = settings.ingestion.metadata_enricher.use_llm
        self.llm = llm
        self._llm_setup_error: str | None = None
        self._last_llm_error: str | None = None
        if self.use_llm and self.llm is None:
            try:
                self.llm = LLMFactory.create(settings)
            except Exception as exc:
                self._llm_setup_error = str(exc)

    def transform(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        enriched: list[Chunk] = []
        for chunk in chunks:
            try:
                rule_metadata = self._rule_metadata(chunk)
                generated = (
                    self._llm_metadata(chunk.text) if self.use_llm else None
                )
                metadata = dict(chunk.metadata)
                metadata.update(generated or rule_metadata)
                if generated is not None:
                    metadata["metadata_enriched_by"] = "llm"
                else:
                    metadata["metadata_enriched_by"] = "rule"
                    if self.use_llm:
                        metadata["metadata_enrichment_fallback_reason"] = (
                            self._last_llm_error or "LLM unavailable"
                        )
                enriched.append(replace(chunk, metadata=metadata))
            except Exception as exc:
                metadata = dict(chunk.metadata)
                metadata["metadata_enriched_by"] = "original"
                metadata["metadata_enrichment_fallback_reason"] = str(exc)
                enriched.append(replace(chunk, metadata=metadata))

        if trace is not None:
            trace.record_stage(
                "metadata_enricher",
                {"chunk_count": len(chunks), "llm_enabled": self.use_llm},
            )
        return enriched

    def _rule_metadata(self, chunk: Chunk) -> dict[str, Any]:
        plain_text = self._plain_text(chunk.text)
        title = self._existing_or_derived_title(chunk, plain_text)
        summary = plain_text[:200].strip() or "No content available."
        tags = self._extract_tags(plain_text)
        return {
            "title": title,
            "summary": summary,
            "tags": tags or ["untitled"],
        }

    def _llm_metadata(self, text: str) -> dict[str, Any] | None:
        self._last_llm_error = self._llm_setup_error
        if self.llm is None:
            return None
        try:
            response = self.llm.chat(
                [{"role": "user", "content": METADATA_PROMPT.format(text=text)}]
            )
            metadata = self._parse_llm_metadata(response)
        except Exception as exc:
            self._last_llm_error = str(exc)
            return None
        self._last_llm_error = None
        return metadata

    @staticmethod
    def _parse_llm_metadata(response: str) -> dict[str, Any]:
        content = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1)
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("LLM metadata must be a JSON object")

        title = payload.get("title")
        summary = payload.get("summary")
        tags = payload.get("tags")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("LLM metadata title must be non-empty")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("LLM metadata summary must be non-empty")
        if not isinstance(tags, list) or not tags:
            raise ValueError("LLM metadata tags must be a non-empty list")
        normalized_tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        if not normalized_tags:
            raise ValueError("LLM metadata tags must contain non-empty strings")
        return {
            "title": title.strip(),
            "summary": summary.strip(),
            "tags": list(dict.fromkeys(normalized_tags)),
        }

    @staticmethod
    def _plain_text(text: str) -> str:
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```(?:\w+)?|```", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _existing_or_derived_title(chunk: Chunk, plain_text: str) -> str:
        existing = chunk.metadata.get("title")
        if isinstance(existing, str) and existing.strip():
            return existing.strip()
        heading = re.search(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", chunk.text, re.MULTILINE)
        if heading:
            return heading.group(1).strip()[:80]
        if plain_text:
            first_sentence = re.split(r"[。！？.!?\n]", plain_text, maxsplit=1)[0]
            return first_sentence.strip()[:80] or plain_text[:80]
        return "Untitled chunk"

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        candidates = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,8}", text
        )
        tags: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.casefold()
            if key not in seen:
                tags.append(candidate)
                seen.add(key)
            if len(tags) == 5:
                break
        return tags
