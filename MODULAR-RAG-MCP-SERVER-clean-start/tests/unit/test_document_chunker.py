import hashlib
from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import DocumentChunker
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.received_text: str | None = None

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        self.received_text = text
        return self.chunks


def settings() -> Settings:
    return load_settings()


def document(text: str, images: list[dict[str, Any]] | None = None) -> Document:
    metadata: dict[str, Any] = {
        "source_path": "docs/sample.pdf",
        "doc_type": "pdf",
        "title": "Sample",
    }
    if images is not None:
        metadata["images"] = images
    return Document(id="doc-1", text=text, metadata=metadata)


@pytest.mark.unit
def test_uses_factory_and_builds_chunks_with_stable_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSplitter(["alpha", "beta"])
    app_settings = settings()
    captured: dict[str, Any] = {}

    def create_splitter(received_settings: Settings) -> FakeSplitter:
        captured["settings"] = received_settings
        return fake

    monkeypatch.setattr(SplitterFactory, "create", create_splitter)
    source = document("alpha middle beta")

    chunks = DocumentChunker(app_settings).split_document(source)

    assert captured["settings"] is app_settings
    assert fake.received_text == source.text
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.start_offset for chunk in chunks] == [0, 13]
    assert [chunk.end_offset for chunk in chunks] == [5, 17]
    assert [chunk.source_ref for chunk in chunks] == ["doc-1", "doc-1"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert chunks[0].id == (
        "doc-1_0000_" + hashlib.sha256(b"alpha").hexdigest()[:8]
    )
    assert len({chunk.id for chunk in chunks}) == 2


@pytest.mark.unit
def test_repeated_split_is_deterministic_and_metadata_is_copied() -> None:
    source = document("first second")
    chunker = DocumentChunker(settings(), splitter=FakeSplitter(["first", "second"]))

    first_run = chunker.split_document(source)
    second_run = chunker.split_document(source)
    first_run[0].metadata["title"] = "Changed"

    assert [chunk.id for chunk in first_run] == [chunk.id for chunk in second_run]
    assert source.metadata["title"] == "Sample"


@pytest.mark.unit
def test_image_references_are_distributed_only_to_matching_chunks() -> None:
    images = [
        {
            "id": "image-a",
            "path": "data/images/image-a.png",
            "page": 1,
            "text_offset": 6,
            "text_length": 16,
            "position": {},
        },
        {
            "id": "image-b",
            "path": "data/images/image-b.png",
            "page": 1,
            "text_offset": 50,
            "text_length": 16,
            "position": {},
        },
    ]
    parts = [
        "first [IMAGE: image-a]",
        "second has no image",
        "third [IMAGE: image-b] [IMAGE: image-a]",
    ]
    source = document("\n\n".join(parts), images)
    chunks = DocumentChunker(settings(), splitter=FakeSplitter(parts)).split_document(
        source
    )

    assert chunks[0].metadata["image_refs"] == ["image-a"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["image-a"]
    assert "images" not in chunks[1].metadata
    assert "image_refs" not in chunks[1].metadata
    assert chunks[2].metadata["image_refs"] == ["image-b", "image-a"]
    assert [image["id"] for image in chunks[2].metadata["images"]] == [
        "image-b",
        "image-a",
    ]


@pytest.mark.unit
def test_overlapping_chunks_keep_original_offsets() -> None:
    source = document("abcdefghij")
    chunker = DocumentChunker(
        settings(), splitter=FakeSplitter(["abcdef", "defghi"])
    )

    chunks = chunker.split_document(source)

    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 6),
        (3, 9),
    ]


@pytest.mark.unit
def test_empty_split_result_returns_no_chunks() -> None:
    chunker = DocumentChunker(settings(), splitter=FakeSplitter([]))

    assert chunker.split_document(document("")) == []


@pytest.mark.unit
def test_chunk_size_configuration_changes_real_split_output() -> None:
    app_settings = settings()
    small_settings = replace(
        app_settings,
        splitter=replace(app_settings.splitter, chunk_size=40, chunk_overlap=5),
    )
    large_settings = replace(
        app_settings,
        splitter=replace(app_settings.splitter, chunk_size=120, chunk_overlap=10),
    )
    source = document("RAG retrieval sentence. " * 20)

    small_chunks = DocumentChunker(small_settings).split_document(source)
    large_chunks = DocumentChunker(large_settings).split_document(source)

    assert len(small_chunks) > len(large_chunks)
    assert all(len(chunk.text) <= 40 for chunk in small_chunks)
    assert all(len(chunk.text) <= 120 for chunk in large_chunks)
