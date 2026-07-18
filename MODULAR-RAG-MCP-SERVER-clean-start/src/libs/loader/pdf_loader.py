"""PDF to Markdown loader with embedded image extraction."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf
from markitdown import MarkItDown
from PIL import Image

from core.types import Document
from libs.loader.base_loader import BaseLoader


logger = logging.getLogger(__name__)


class PdfLoaderError(RuntimeError):
    """Raised when PDF text cannot be parsed."""


class PdfLoader(BaseLoader):
    def __init__(
        self,
        images_root: str | Path = "data/images",
        converter: Any | None = None,
    ) -> None:
        self.images_root = Path(images_root)
        self._converter = converter or MarkItDown(enable_plugins=False)

    def load(self, path: str) -> Document:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"PDF file not found: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {source}")

        document_hash = self._compute_sha256(source)
        try:
            result = self._converter.convert(str(source))
            text = result.text_content
        except Exception as exc:
            raise PdfLoaderError(f"Failed to parse PDF '{source}': {exc}") from exc
        if not isinstance(text, str):
            raise PdfLoaderError("MarkItDown returned non-string text content")

        try:
            images = self._extract_images(source, document_hash)
        except Exception as exc:
            logger.warning("Failed to extract images from %s: %s", source, exc)
            images = []
        text = self._append_image_placeholders(text.strip(), images)

        return Document(
            id=document_hash,
            text=text,
            metadata={
                "source_path": str(source.resolve()),
                "doc_type": "pdf",
                "title": source.stem,
                "images": images,
            },
        )

    def _extract_images(
        self, source: Path, document_hash: str
    ) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        output_dir = self.images_root / document_hash
        with pymupdf.open(source) as pdf:
            sequence = 0
            for page in pdf:
                blocks = page.get_text("dict")["blocks"]
                for block_index, block in enumerate(blocks):
                    if block.get("type") != 1 or not block.get("image"):
                        continue
                    sequence += 1
                    image_id = f"{document_hash}_{page.number + 1}_{sequence}"
                    image_path = output_dir / f"{image_id}.png"
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        self._save_image(block["image"], image_path)
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract image %s from %s: %s",
                            image_id,
                            source,
                            exc,
                        )
                        continue
                    images.append(
                        {
                            "id": image_id,
                            "path": str(image_path),
                            "page": page.number + 1,
                            "position": self._position(block.get("bbox")),
                            "_next_text": self._next_text(blocks, block_index),
                        }
                    )
        return images

    @staticmethod
    def _save_image(image_data: bytes, image_path: Path) -> None:
        with Image.open(BytesIO(image_data)) as image:
            image.save(image_path, format="PNG")

    @staticmethod
    def _position(bbox: Any) -> dict[str, float]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return {}
        return {
            "x0": float(bbox[0]),
            "y0": float(bbox[1]),
            "x1": float(bbox[2]),
            "y1": float(bbox[3]),
        }

    @staticmethod
    def _next_text(blocks: list[dict[str, Any]], block_index: int) -> str | None:
        for block in blocks[block_index + 1 :]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        return None

    @staticmethod
    def _append_image_placeholders(
        text: str, images: list[dict[str, Any]]
    ) -> str:
        for image in images:
            placeholder = f"[IMAGE: {image['id']}]"
            anchor = image.pop("_next_text", None)
            anchor_offset = text.find(anchor) if anchor else -1
            if anchor_offset >= 0:
                prefix = text[:anchor_offset].rstrip()
                suffix = text[anchor_offset:].lstrip()
            else:
                prefix = text.rstrip()
                suffix = ""
            separator = "\n\n" if prefix else ""
            image["text_offset"] = len(prefix) + len(separator)
            image["text_length"] = len(placeholder)
            trailing = f"\n\n{suffix}" if suffix else ""
            text = f"{prefix}{separator}{placeholder}{trailing}"
        return text

    @staticmethod
    def _compute_sha256(source: Path) -> str:
        digest = hashlib.sha256()
        with source.open("rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
