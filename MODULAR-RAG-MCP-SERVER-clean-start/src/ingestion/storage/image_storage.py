"""Filesystem image storage backed by a SQLite lookup index."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


class ImageStorage:
    def __init__(
        self,
        images_root: str | Path = "data/images",
        db_path: str | Path = "data/db/image_index.db",
    ) -> None:
        self.images_root = Path(images_root)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def save(
        self,
        image_id: str,
        image_data: bytes,
        *,
        collection: str,
        doc_hash: str | None = None,
        page_num: int | None = None,
    ) -> Path:
        self._validate_path_part(image_id, "image_id")
        self._validate_path_part(collection, "collection")
        if not isinstance(image_data, bytes) or not image_data:
            raise ValueError("image_data must be non-empty bytes")
        if doc_hash is not None and (
            not isinstance(doc_hash, str) or not doc_hash.strip()
        ):
            raise ValueError("doc_hash must be a non-empty string when provided")
        if page_num is not None and (
            not isinstance(page_num, int)
            or isinstance(page_num, bool)
            or page_num < 0
        ):
            raise ValueError("page_num must be a non-negative integer")

        output_dir = self.images_root / collection
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{image_id}.png"
        self._write_atomic(image_path, image_data)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_index (
                    image_id, file_path, collection, doc_hash, page_num
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num
                """,
                (image_id, str(image_path), collection, doc_hash, page_num),
            )
        return image_path

    def get_path(self, image_id: str) -> Path | None:
        self._validate_path_part(image_id, "image_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        return Path(row[0]) if row is not None else None

    def list_by_collection(self, collection: str) -> list[dict[str, Any]]:
        self._validate_path_part(collection, "collection")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT image_id, file_path, collection, doc_hash, page_num,
                       created_at
                FROM image_index
                WHERE collection = ?
                ORDER BY image_id
                """,
                (collection,),
            ).fetchall()
        return [
            {
                "image_id": row[0],
                "file_path": Path(row[1]),
                "collection": row[2],
                "doc_hash": row[3],
                "page_num": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection "
                "ON image_index(collection)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash "
                "ON image_index(doc_hash)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _write_atomic(image_path: Path, image_data: bytes) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=image_path.parent,
                prefix=f".{image_path.stem}-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(image_data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, image_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_path_part(value: str, field_name: str) -> None:
        if (
            not isinstance(value, str)
            or not value.strip()
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(f"{field_name} must be a safe non-empty name")
