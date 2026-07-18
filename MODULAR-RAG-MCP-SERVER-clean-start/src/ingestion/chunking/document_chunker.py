"""Adapt plain text splitter output into shared Chunk contracts."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


IMAGE_PLACEHOLDER = re.compile(r"\[IMAGE:\s*([^\]]+?)\]")


class DocumentChunker:
    def __init__(
        self, settings: Settings, splitter: BaseSplitter | None = None
    ) -> None:
        self.settings = settings
        self._splitter = splitter or SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        chunk_texts = self._splitter.split_text(document.text)
        chunks: list[Chunk] = []
        search_start = 0
        for index, chunk_text in enumerate(chunk_texts):
            if not isinstance(chunk_text, str) or not chunk_text:
                raise ValueError(f"Splitter returned invalid chunk at index {index}")
            start_offset = document.text.find(chunk_text, search_start)
            if start_offset < 0:
                raise ValueError(
                    f"Chunk at index {index} cannot be located in the source document"
                )
            end_offset = start_offset + len(chunk_text)
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, chunk_text),
                    text=chunk_text,
                    metadata=self._inherit_metadata(document, index, chunk_text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
            search_start = start_offset + 1
        return chunks

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    @staticmethod
    def _inherit_metadata(
        document: Document, chunk_index: int, chunk_text: str
    ) -> dict[str, Any]:
        metadata = deepcopy(document.metadata)
        document_images = metadata.pop("images", [])
        metadata.pop("image_refs", None)
        metadata["chunk_index"] = chunk_index

        image_refs = [match.group(1).strip() for match in IMAGE_PLACEHOLDER.finditer(chunk_text)]
        if not image_refs:
            return metadata

        images_by_id = {image["id"]: image for image in document_images}
        selected_images: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for image_id in image_refs:
            image = images_by_id.get(image_id)
            if image is not None and image_id not in seen_ids:
                selected_images.append(image)
                seen_ids.add(image_id)
        metadata["image_refs"] = image_refs
        metadata["images"] = selected_images
        return metadata
