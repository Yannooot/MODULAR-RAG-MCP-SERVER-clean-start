"""Shared data contracts for ingestion, retrieval, and MCP layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any


class _SerializableContract:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def _validate_common(self) -> None:
        identifier = getattr(self, "id")
        text = getattr(self, "text")
        metadata = getattr(self, "metadata")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("id must be a non-empty string")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        _validate_metadata(metadata)


@dataclass
class Document(_SerializableContract):
    id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass
class Chunk(_SerializableContract):
    id: str
    text: str
    metadata: dict[str, Any]
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def __post_init__(self) -> None:
        self._validate_common()
        if (
            not _is_integer(self.start_offset)
            or not _is_integer(self.end_offset)
            or self.start_offset < 0
            or self.end_offset < self.start_offset
        ):
            raise ValueError(
                "offsets must be non-negative integers with end_offset >= start_offset"
            )
        if self.source_ref is not None and (
            not isinstance(self.source_ref, str) or not self.source_ref.strip()
        ):
            raise ValueError("source_ref must be a non-empty string when provided")


@dataclass
class ChunkRecord(_SerializableContract):
    id: str
    text: str
    metadata: dict[str, Any]
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self._validate_common()
        if self.dense_vector is not None and (
            not isinstance(self.dense_vector, list)
            or any(not _is_number(value) for value in self.dense_vector)
        ):
            raise ValueError("dense_vector must contain only numeric values")
        if self.sparse_vector is not None and (
            not isinstance(self.sparse_vector, dict)
            or any(
                not isinstance(term, str) or not _is_number(weight)
                for term, weight in self.sparse_vector.items()
            )
        ):
            raise ValueError("sparse_vector must map string terms to numeric weights")


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("metadata.source_path must be a non-empty string")

    images = metadata.get("images")
    if images is None:
        return
    if not isinstance(images, list):
        raise ValueError("metadata.images must be a list")
    for index, image in enumerate(images):
        _validate_image_reference(image, index)


def _validate_image_reference(image: Any, index: int) -> None:
    prefix = f"metadata.images[{index}]"
    if not isinstance(image, dict):
        raise ValueError(f"{prefix} must be a mapping")
    for field_name in ("id", "path"):
        value = image.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}.{field_name} must be a non-empty string")
    for field_name in ("text_offset", "text_length"):
        value = image.get(field_name)
        if not _is_integer(value) or value < 0:
            raise ValueError(f"{prefix}.{field_name} must be a non-negative integer")
    page = image.get("page")
    if page is not None and (not _is_integer(page) or page < 0):
        raise ValueError(f"{prefix}.page must be a non-negative integer")
    position = image.get("position")
    if position is not None and not isinstance(position, dict):
        raise ValueError(f"{prefix}.position must be a mapping")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
