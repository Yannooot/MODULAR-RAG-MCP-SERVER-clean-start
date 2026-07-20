import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage


@pytest.mark.unit
def test_default_paths_are_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    storage = ImageStorage()

    assert storage.images_root == Path("data/images")
    assert storage.db_path == Path("data/db/image_index.db")
    assert storage.db_path.is_file()


@pytest.mark.unit
def test_save_writes_file_and_persists_lookup(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    db_path = tmp_path / "db" / "images.db"
    storage = ImageStorage(images_root, db_path)

    saved = storage.save(
        "image-1",
        b"png bytes",
        collection="manuals",
        doc_hash="a" * 64,
        page_num=3,
    )
    reopened = ImageStorage(images_root, db_path)

    assert saved == images_root / "manuals" / "image-1.png"
    assert saved.read_bytes() == b"png bytes"
    assert reopened.get_path("image-1") == saved

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT collection, doc_hash, page_num FROM image_index "
            "WHERE image_id = 'image-1'"
        ).fetchone()
    assert row == ("manuals", "a" * 64, 3)


@pytest.mark.unit
def test_list_by_collection_is_ordered_and_isolated(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "images.db")
    storage.save("image-b", b"b", collection="alpha")
    storage.save("image-a", b"a", collection="alpha")
    storage.save("image-c", b"c", collection="beta")

    results = storage.list_by_collection("alpha")

    assert [item["image_id"] for item in results] == ["image-a", "image-b"]
    assert all(item["collection"] == "alpha" for item in results)
    assert all(isinstance(item["file_path"], Path) for item in results)


@pytest.mark.unit
def test_existing_image_id_is_updated_without_duplicate_row(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "images.db")
    storage.save("same", b"old", collection="first", page_num=1)

    updated = storage.save("same", b"new", collection="second", page_num=2)

    assert updated.read_bytes() == b"new"
    assert storage.get_path("same") == updated
    with sqlite3.connect(storage.db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM image_index WHERE image_id = 'same'"
        ).fetchone()[0]
    assert count == 1


@pytest.mark.unit
def test_database_uses_wal_and_accepts_concurrent_writes(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "images.db")

    def save_image(index: int) -> Path:
        return storage.save(
            f"image-{index}",
            f"content-{index}".encode(),
            collection="concurrent",
            doc_hash=f"{index:064x}",
            page_num=index,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(save_image, range(24)))

    with sqlite3.connect(storage.db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode == "wal"
    assert all(path.is_file() for path in paths)
    assert len(storage.list_by_collection("concurrent")) == 24


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "../escape", "a/b", "a\\b"])
def test_image_id_rejects_empty_or_unsafe_values(
    tmp_path: Path, value: str
) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "images.db")

    with pytest.raises(ValueError, match="image_id"):
        storage.save(value, b"data", collection="safe")


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "../escape", "a/b", "a\\b"])
def test_collection_rejects_empty_or_unsafe_values(
    tmp_path: Path, value: str
) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "images.db")

    with pytest.raises(ValueError, match="collection"):
        storage.save("image", b"data", collection=value)


@pytest.mark.unit
def test_empty_image_data_and_invalid_page_are_rejected(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path / "images", tmp_path / "images.db")

    with pytest.raises(ValueError, match="image_data"):
        storage.save("image", b"", collection="safe")
    with pytest.raises(ValueError, match="page_num"):
        storage.save("image", b"data", collection="safe", page_num=True)
