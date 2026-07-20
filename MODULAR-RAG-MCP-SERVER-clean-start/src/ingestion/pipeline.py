"""Serial orchestration for the offline ingestion workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, dict[str, Any]], None]


class PipelineError(RuntimeError):
    """Raised when an ingestion stage cannot complete."""


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        integrity_checker: FileIntegrityChecker | None = None,
        loader: BaseLoader | None = None,
        chunker: DocumentChunker | None = None,
        transforms: Sequence[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        bm25_indexer: BM25Indexer | None = None,
        vector_upserter: VectorUpserter | None = None,
        image_storage: ImageStorage | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self.image_storage = image_storage or ImageStorage()
        self.loader = loader or PdfLoader(images_root=self.image_storage.images_root)
        self.chunker = chunker or DocumentChunker(settings)
        self.transforms = list(transforms) if transforms is not None else [
            ChunkRefiner(settings),
            MetadataEnricher(settings),
            ImageCaptioner(settings),
        ]
        self.batch_processor = batch_processor or BatchProcessor(settings)
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.on_progress = on_progress

    def run(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        force: bool = False,
        trace: TraceContext | None = None,
    ) -> dict[str, Any]:
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")
        collection_name = (
            self.settings.vector_store.collection_name
            if collection is None
            else collection
        )
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection must be a non-empty string")

        source = Path(path)
        context = trace or TraceContext()
        file_hash: str | None = None
        stage = "integrity"
        try:
            file_hash = self.integrity_checker.compute_sha256(str(source))
            skipped = not force and self.integrity_checker.should_skip(file_hash)
            self._report(
                context,
                "integrity",
                {"file_hash": file_hash, "skipped": skipped, "force": force},
            )
            if skipped:
                return {
                    "status": "skipped",
                    "file_hash": file_hash,
                    "source_path": str(source),
                    "chunk_count": 0,
                    "image_count": 0,
                    "vector_ids": [],
                    "trace_id": context.trace_id,
                }

            stage = "load"
            document = self.loader.load(str(source))
            if not isinstance(document, Document):
                raise TypeError("loader must return a Document")
            self._report(context, "load", {"document_id": document.id})

            stage = "split"
            chunks = self.chunker.split_document(document)
            self._report(context, "split", {"chunk_count": len(chunks)})

            stage = "transform"
            for transform in self.transforms:
                chunks = transform.transform(chunks, context)
            self._report(
                context,
                "transform",
                {
                    "chunk_count": len(chunks),
                    "transform_count": len(self.transforms),
                },
            )

            stage = "encode"
            records = self.batch_processor.process(chunks, context)
            self._report(context, "encode", {"record_count": len(records)})

            stage = "store"
            source_path = document.metadata["source_path"]
            self.bm25_indexer.remove_document(source_path)
            self.bm25_indexer.update(records)
            vector_ids = self.vector_upserter.upsert(records, context)
            image_count = self._store_images(document, collection_name)
            self._report(
                context,
                "store",
                {
                    "record_count": len(records),
                    "image_count": image_count,
                },
            )

            self.integrity_checker.mark_success(
                file_hash,
                str(source),
                file_size=source.stat().st_size,
                chunk_count=len(records),
            )
            result = {
                "status": "success",
                "file_hash": file_hash,
                "source_path": str(source),
                "chunk_count": len(records),
                "image_count": image_count,
                "vector_ids": vector_ids,
                "trace_id": context.trace_id,
            }
            self._report(context, "complete", result)
            return result
        except Exception as exc:
            if file_hash is not None:
                try:
                    self.integrity_checker.mark_failed(file_hash, str(exc))
                except Exception:
                    logger.exception("Failed to record ingestion failure")
            self._report(
                context,
                "failed",
                {"stage": stage, "error": str(exc)},
            )
            raise PipelineError(f"{stage} stage failed: {exc}") from exc

    def _store_images(self, document: Document, collection: str) -> int:
        images = document.metadata.get("images", [])
        if not isinstance(images, list):
            raise ValueError("document metadata.images must be a list")
        stored = 0
        for image in images:
            if not isinstance(image, dict):
                raise ValueError("document image metadata must be a mapping")
            image_id = image.get("id")
            image_path = image.get("path")
            if not isinstance(image_path, str) or not Path(image_path).is_file():
                raise FileNotFoundError(f"image file not found: {image_path}")
            self.image_storage.save(
                image_id,
                Path(image_path).read_bytes(),
                collection=collection,
                doc_hash=document.id,
                page_num=image.get("page"),
            )
            stored += 1
        return stored

    def _report(
        self,
        trace: TraceContext,
        stage: str,
        details: dict[str, Any],
    ) -> None:
        logger.info("Ingestion stage %s: %s", stage, details)
        trace.record_stage(f"pipeline_{stage}", details)
        if self.on_progress is not None:
            self.on_progress(stage, details)
