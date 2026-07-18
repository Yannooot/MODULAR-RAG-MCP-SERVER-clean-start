"""File hashing and SQLite-backed ingestion history."""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class FileIntegrityChecker(ABC):
    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """Return the SHA256 digest of a file."""
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return whether a file hash has already completed successfully."""
        raise NotImplementedError

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Persist a successful ingestion result."""
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Persist a failed ingestion result so it can be retried."""
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    def __init__(
        self, db_path: str | Path = "data/db/ingestion_history.db"
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def compute_sha256(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        self._validate_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM ingestion_history "
                "WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
        return row is not None

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        self._validate_hash(file_hash)
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        self._validate_count(file_size, "file_size")
        self._validate_count(chunk_count, "chunk_count")
        if file_size is None:
            source = Path(file_path)
            file_size = source.stat().st_size if source.is_file() else None

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, processed_at,
                    error_msg, chunk_count
                ) VALUES (?, ?, ?, 'success', CURRENT_TIMESTAMP, NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    status = 'success',
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = NULL,
                    chunk_count = excluded.chunk_count
                """,
                (file_hash, file_path, file_size, chunk_count),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        self._validate_hash(file_hash)
        if not isinstance(error_msg, str) or not error_msg.strip():
            raise ValueError("error_msg must be a non-empty string")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, processed_at, error_msg
                ) VALUES (?, '', 'failed', CURRENT_TIMESTAMP, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status = 'failed',
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = excluded.error_msg
                """,
                (file_hash, error_msg),
            )

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(
                        status IN ('success', 'failed', 'processing')
                    ),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_status "
                "ON ingestion_history(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at "
                "ON ingestion_history(processed_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _validate_hash(file_hash: str) -> None:
        if not isinstance(file_hash, str) or not file_hash.strip():
            raise ValueError("file_hash must be a non-empty string")

    @staticmethod
    def _validate_count(value: int | None, field_name: str) -> None:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{field_name} must be a non-negative integer")
