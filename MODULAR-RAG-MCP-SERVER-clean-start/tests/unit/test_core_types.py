import json

import pytest

from core.types import Chunk, ChunkRecord, Document


@pytest.mark.unit
def test_document_has_stable_dict_and_json_shape() -> None:
    document = Document(
        id="doc-1",
        text="你好 [IMAGE: image-1]",
        metadata={"source_path": "docs/sample.pdf", "title": "Sample"},
    )

    assert list(document.to_dict()) == ["id", "text", "metadata"]
    assert document.to_dict() == {
        "id": "doc-1",
        "text": "你好 [IMAGE: image-1]",
        "metadata": {"source_path": "docs/sample.pdf", "title": "Sample"},
    }
    assert json.loads(document.to_json()) == document.to_dict()


@pytest.mark.unit
def test_chunk_has_stable_serialized_offsets_and_source_reference() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="chunk text",
        metadata={"source_path": "docs/sample.pdf", "chunk_index": 0},
        start_offset=10,
        end_offset=20,
        source_ref="doc-1",
    )

    assert chunk.to_dict() == {
        "id": "chunk-1",
        "text": "chunk text",
        "metadata": {"source_path": "docs/sample.pdf", "chunk_index": 0},
        "start_offset": 10,
        "end_offset": 20,
        "source_ref": "doc-1",
    }


@pytest.mark.unit
def test_chunk_record_serializes_optional_vector_fields() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="chunk text",
        metadata={"source_path": "docs/sample.pdf"},
        dense_vector=[0.1, 0.2],
        sparse_vector={"rag": 0.8},
    )

    assert record.to_dict() == {
        "id": "chunk-1",
        "text": "chunk text",
        "metadata": {"source_path": "docs/sample.pdf"},
        "dense_vector": [0.1, 0.2],
        "sparse_vector": {"rag": 0.8},
    }
    assert json.loads(record.to_json()) == record.to_dict()


@pytest.mark.unit
def test_optional_vectors_are_stable_null_fields() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="chunk text",
        metadata={"source_path": "docs/sample.pdf"},
    )

    assert record.to_dict()["dense_vector"] is None
    assert record.to_dict()["sparse_vector"] is None


@pytest.mark.unit
def test_image_metadata_contract_allows_optional_page_and_position() -> None:
    images = [
        {
            "id": "hash_1_0",
            "path": "data/images/default/hash_1_0.png",
            "page": 1,
            "text_offset": 6,
            "text_length": 17,
            "position": {"x": 10, "y": 20},
        },
        {
            "id": "hash_1_1",
            "path": "data/images/default/hash_1_1.png",
            "text_offset": 30,
            "text_length": 17,
        },
    ]

    document = Document(
        id="doc-1",
        text="text [IMAGE: hash_1_0]",
        metadata={"source_path": "docs/sample.pdf", "images": images},
    )

    assert document.to_dict()["metadata"]["images"] == images


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"source_path": ""},
        {"source_path": "docs/sample.pdf", "images": "not-a-list"},
        {
            "source_path": "docs/sample.pdf",
            "images": [{"id": "image-1", "text_offset": 0, "text_length": 1}],
        },
        {
            "source_path": "docs/sample.pdf",
            "images": [
                {
                    "id": "image-1",
                    "path": "image.png",
                    "page": "one",
                    "text_offset": 0,
                    "text_length": 1,
                }
            ],
        },
    ],
)
def test_invalid_metadata_has_readable_error(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="metadata"):
        Document(id="doc-1", text="text", metadata=metadata)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start_offset", "end_offset"),
    [(-1, 2), (3, 2), (True, 2)],
)
def test_invalid_chunk_offsets_have_readable_error(
    start_offset: int, end_offset: int
) -> None:
    with pytest.raises(ValueError, match="offset"):
        Chunk(
            id="chunk-1",
            text="text",
            metadata={"source_path": "docs/sample.pdf"},
            start_offset=start_offset,
            end_offset=end_offset,
        )
