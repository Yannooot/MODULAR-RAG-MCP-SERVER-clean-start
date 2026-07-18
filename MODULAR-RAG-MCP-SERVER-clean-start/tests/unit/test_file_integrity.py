import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker


@pytest.mark.unit
def test_default_database_path_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    checker = SQLiteIntegrityChecker()

    assert checker.db_path == Path("data/db/ingestion_history.db")
    assert checker.db_path.is_file()


@pytest.mark.unit
def test_compute_sha256_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    content = b"modular rag\n"
    source.write_bytes(content)
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    first = checker.compute_sha256(str(source))
    second = checker.compute_sha256(str(source))

    assert first == second
    assert first == hashlib.sha256(content).hexdigest()


@pytest.mark.unit
def test_only_success_status_is_skipped_and_fields_are_persisted(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db" / "history.db"
    checker = SQLiteIntegrityChecker(db_path)
    file_hash = "a" * 64

    checker.mark_success(
        file_hash,
        "docs/sample.pdf",
        file_size=123,
        chunk_count=7,
    )

    assert checker.should_skip(file_hash) is True
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT file_path, file_size, status, error_msg, chunk_count "
            "FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
    assert row == ("docs/sample.pdf", 123, "success", None, 7)

    checker.mark_failed(file_hash, "embedding failed")

    assert checker.should_skip(file_hash) is False
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT file_path, status, error_msg FROM ingestion_history "
            "WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
    assert row == ("docs/sample.pdf", "failed", "embedding failed")


@pytest.mark.unit
def test_failed_hash_can_be_recorded_before_success(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "b" * 64

    checker.mark_failed(file_hash, "parse failed")

    assert checker.should_skip(file_hash) is False


@pytest.mark.unit
def test_database_uses_wal_and_accepts_concurrent_writes(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    def write_record(index: int) -> None:
        checker.mark_success(
            f"{index:064x}",
            f"docs/{index}.pdf",
            file_size=index,
            chunk_count=index,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_record, range(24)))

    with sqlite3.connect(checker.db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        count = connection.execute(
            "SELECT COUNT(*) FROM ingestion_history WHERE status = 'success'"
        ).fetchone()[0]
    assert journal_mode == "wal"
    assert count == 24


@pytest.mark.unit
def test_abstract_checker_requires_all_operations() -> None:
    class IncompleteChecker(FileIntegrityChecker):
        pass

    with pytest.raises(TypeError):
        IncompleteChecker()
