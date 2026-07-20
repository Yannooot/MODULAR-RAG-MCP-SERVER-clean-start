from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from PIL import Image

from core.settings import load_settings
from core.types import Chunk, ChunkRecord, Document
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.pipeline import IngestionPipeline, PipelineError
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


class FakeLoader:
    def __init__(self, document: Document, events: list[str]) -> None:
        self.document = document
        self.events = events

    def load(self, path: str) -> Document:
        self.events.append("load")
        return self.document


class FailingLoader:
    def load(self, path: str) -> Document:
        raise RuntimeError("broken pdf")


class FakeChunker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def split_document(self, document: Document) -> list[Chunk]:
        self.events.append("split")
        return [
            Chunk(
                id="chunk-0",
                text=document.text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                },
                start_offset=0,
                end_offset=len(document.text),
                source_ref=document.id,
            )
        ]


class FakeTransform:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def transform(self, chunks: list[Chunk], trace: Any = None) -> list[Chunk]:
        self.events.append(self.name)
        return chunks


class FakeBatchProcessor:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def process(
        self, chunks: list[Chunk], trace: Any = None
    ) -> list[ChunkRecord]:
        self.events.append("encode")
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                dense_vector=[1.0, 0.0],
                sparse_vector={"pipeline": 1.0},
            )
            for chunk in chunks
        ]


class FakeBM25Indexer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def remove_document(self, source_path: str) -> int:
        self.events.append("bm25.remove")
        return 0

    def update(self, records: list[ChunkRecord]) -> None:
        self.events.append("bm25.update")


class FakeVectorUpserter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def upsert(self, records: list[ChunkRecord], trace: Any = None) -> list[str]:
        self.events.append("vector.upsert")
        return [f"vector-{record.id}" for record in records]


class MemoryVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, records: list[dict[str, Any]], trace: Any = None) -> None:
        for record in records:
            self.records[record["id"]] = record


class DeterministicEmbedding:
    def embed(
        self, texts: list[str], trace: Any = None
    ) -> list[list[float]]:
        return [[1.0, float(len(text)) / 1000] for text in texts]


def document(source_path: Path, images: list[dict[str, Any]] | None = None) -> Document:
    metadata: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "doc_type": "pdf",
        "title": "Pipeline",
    }
    if images is not None:
        metadata["images"] = images
    return Document(id="doc-hash", text="pipeline text", metadata=metadata)


def create_complex_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    image_buffer = BytesIO()
    Image.new("RGB", (40, 20), color="red").save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    for page_index in range(3):
        page = pdf.new_page()
        lines = [
            f"Chapter {index}: Modular RAG architecture and storage design"
            for index in range(page_index * 3 + 1, min(page_index * 3 + 4, 9))
        ]
        lines.extend(
            f"Table {index} | Component | Responsibility | Status"
            for index in range(page_index * 2 + 1, min(page_index * 2 + 3, 6))
        )
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 250),
            "\n".join(lines),
            fontsize=11,
        )
        page.insert_image(
            pymupdf.Rect(72, 280, 272, 380), stream=image_bytes
        )
        page.insert_text((72, 410), f"Diagram explanation for page {page_index + 1}")
    pdf.save(path)
    pdf.close()


@pytest.mark.integration
def test_pipeline_runs_stages_in_order_and_reports_progress(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    events: list[str] = []
    progress: list[str] = []
    integrity = SQLiteIntegrityChecker(tmp_path / "history.db")
    pipeline = IngestionPipeline(
        load_settings(),
        integrity_checker=integrity,
        loader=FakeLoader(document(source), events),
        chunker=FakeChunker(events),
        transforms=[
            FakeTransform("refine", events),
            FakeTransform("metadata", events),
            FakeTransform("caption", events),
        ],
        batch_processor=FakeBatchProcessor(events),
        bm25_indexer=FakeBM25Indexer(events),
        vector_upserter=FakeVectorUpserter(events),
        image_storage=ImageStorage(tmp_path / "images", tmp_path / "images.db"),
        on_progress=lambda stage, details: progress.append(stage),
    )

    result = pipeline.run(source, collection="manuals")

    assert result["status"] == "success"
    assert result["chunk_count"] == 1
    assert result["vector_ids"] == ["vector-chunk-0"]
    assert events == [
        "load",
        "split",
        "refine",
        "metadata",
        "caption",
        "encode",
        "bm25.remove",
        "bm25.update",
        "vector.upsert",
    ]
    assert progress == [
        "integrity",
        "load",
        "split",
        "transform",
        "encode",
        "store",
        "complete",
    ]
    assert integrity.should_skip(result["file_hash"]) is True


@pytest.mark.integration
def test_unchanged_file_is_skipped_unless_force_is_true(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    events: list[str] = []
    integrity = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = integrity.compute_sha256(str(source))
    integrity.mark_success(file_hash, str(source), chunk_count=1)
    pipeline = IngestionPipeline(
        load_settings(),
        integrity_checker=integrity,
        loader=FakeLoader(document(source), events),
        chunker=FakeChunker(events),
        transforms=[],
        batch_processor=FakeBatchProcessor(events),
        bm25_indexer=FakeBM25Indexer(events),
        vector_upserter=FakeVectorUpserter(events),
        image_storage=ImageStorage(tmp_path / "images", tmp_path / "images.db"),
    )

    skipped = pipeline.run(source)
    assert events == []
    forced = pipeline.run(source, force=True)

    assert skipped["status"] == "skipped"
    assert forced["status"] == "success"
    assert events[0] == "load"


@pytest.mark.integration
def test_failure_names_stage_and_marks_ingestion_failed(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    integrity = SQLiteIntegrityChecker(tmp_path / "history.db")
    events: list[str] = []
    pipeline = IngestionPipeline(
        load_settings(),
        integrity_checker=integrity,
        loader=FailingLoader(),
        chunker=FakeChunker(events),
        transforms=[],
        batch_processor=FakeBatchProcessor(events),
        bm25_indexer=FakeBM25Indexer(events),
        vector_upserter=FakeVectorUpserter(events),
        image_storage=ImageStorage(tmp_path / "images", tmp_path / "images.db"),
    )

    with pytest.raises(PipelineError, match="load.*broken pdf"):
        pipeline.run(source)

    file_hash = integrity.compute_sha256(str(source))
    assert integrity.should_skip(file_hash) is False


@pytest.mark.integration
def test_pipeline_creates_local_storage_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    extracted = tmp_path / "extracted.png"
    extracted.write_bytes(b"image bytes")
    image = {
        "id": "image-1",
        "path": str(extracted),
        "page": 1,
        "position": {},
        "text_offset": 0,
        "text_length": 0,
    }
    events: list[str] = []
    bm25_path = tmp_path / "db" / "bm25" / "index.pkl"
    image_storage = ImageStorage(
        tmp_path / "images", tmp_path / "db" / "image_index.db"
    )
    vector_store = MemoryVectorStore()
    pipeline = IngestionPipeline(
        load_settings(),
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "db" / "history.db"),
        loader=FakeLoader(document(source, [image]), events),
        chunker=FakeChunker(events),
        transforms=[],
        batch_processor=FakeBatchProcessor(events),
        bm25_indexer=BM25Indexer(bm25_path),
        vector_upserter=VectorUpserter(load_settings(), vector_store),
        image_storage=image_storage,
    )

    result = pipeline.run(source, collection="manuals")

    assert result["image_count"] == 1
    assert bm25_path.is_file()
    assert BM25Indexer(bm25_path).query(["pipeline"])[0]["text"] == "pipeline text"
    assert len(vector_store.records) == 1
    stored_image = image_storage.get_path("image-1")
    assert stored_image == tmp_path / "images" / "manuals" / "image-1.png"
    assert stored_image.read_bytes() == b"image bytes"


@pytest.mark.integration
def test_generated_pdf_runs_through_real_local_components(tmp_path: Path) -> None:
    source = tmp_path / "complex_technical_doc.pdf"
    create_complex_pdf(source)
    base_settings = load_settings()
    settings = replace(
        base_settings,
        vision_llm=replace(base_settings.vision_llm, enabled=False),
        splitter=replace(
            base_settings.splitter, chunk_size=300, chunk_overlap=30
        ),
        vector_store=replace(
            base_settings.vector_store,
            persist_path=str(tmp_path / "db" / "chroma"),
            collection_name="technical",
        ),
        ingestion=replace(
            base_settings.ingestion,
            chunk_refiner=replace(
                base_settings.ingestion.chunk_refiner, use_llm=False
            ),
            metadata_enricher=replace(
                base_settings.ingestion.metadata_enricher, use_llm=False
            ),
        ),
    )
    batch_processor = BatchProcessor(
        settings,
        dense_encoder=DenseEncoder(settings, DeterministicEmbedding()),
        sparse_encoder=SparseEncoder(settings),
    )
    bm25_path = tmp_path / "db" / "bm25" / "index.pkl"
    image_storage = ImageStorage(
        tmp_path / "images", tmp_path / "db" / "image_index.db"
    )
    pipeline = IngestionPipeline(
        settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "db" / "history.db"),
        batch_processor=batch_processor,
        bm25_indexer=BM25Indexer(bm25_path),
        image_storage=image_storage,
    )

    result = pipeline.run(source)

    assert result["status"] == "success"
    assert result["chunk_count"] > 0
    assert result["image_count"] == 3
    assert bm25_path.is_file()
    assert (tmp_path / "db" / "chroma").is_dir()
    assert len(image_storage.list_by_collection("technical")) == 3
    assert ChromaStore(settings).query([1.0, 0.1], top_k=1)
