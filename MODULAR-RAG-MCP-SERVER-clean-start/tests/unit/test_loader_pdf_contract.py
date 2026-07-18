import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from core.types import Document
from libs.loader.base_loader import BaseLoader
from libs.loader.pdf_loader import PdfLoader
from PIL import Image
import pymupdf


def create_pdf(path: Path, *, with_image: bool = False) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Simple PDF document for RAG")
    if with_image:
        buffer = BytesIO()
        Image.new("RGB", (20, 10), color="red").save(buffer, format="PNG")
        page.insert_image(
            pymupdf.Rect(72, 100, 172, 150), stream=buffer.getvalue()
        )
        page.insert_text((72, 180), "Text after image")
    document.save(path)
    document.close()


@pytest.mark.unit
def test_pdf_loader_returns_document_with_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "simple.pdf"
    create_pdf(pdf_path)

    document = PdfLoader(images_root=tmp_path / "images").load(str(pdf_path))

    assert isinstance(document, Document)
    assert "Simple PDF document for RAG" in document.text
    assert document.id and len(document.id) == 64
    assert document.metadata == {
        "source_path": str(pdf_path.resolve()),
        "doc_type": "pdf",
        "title": "simple",
        "images": [],
    }


@pytest.mark.unit
def test_pdf_loader_extracts_images_and_records_placeholder_offsets(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "with_images.pdf"
    create_pdf(pdf_path, with_image=True)

    document = PdfLoader(images_root=tmp_path / "images").load(str(pdf_path))

    assert len(document.metadata["images"]) == 1
    image = document.metadata["images"][0]
    placeholder = f'[IMAGE: {image["id"]}]'
    assert document.text[image["text_offset"] : image["text_offset"] + image["text_length"]] == placeholder
    assert Path(image["path"]).is_file()
    assert document.text.index(placeholder) < document.text.index("Text after image")
    assert image["page"] == 1
    assert image["position"] == {
        "x0": pytest.approx(72.0),
        "y0": pytest.approx(100.0),
        "x1": pytest.approx(172.0),
        "y1": pytest.approx(150.0),
    }


@pytest.mark.unit
def test_image_extraction_failure_keeps_parsed_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pdf_path = tmp_path / "simple.pdf"
    create_pdf(pdf_path)
    loader = PdfLoader(images_root=tmp_path / "images")

    def fail_extraction(path: Path, document_hash: str) -> list[dict[str, Any]]:
        raise RuntimeError("image extraction unavailable")

    monkeypatch.setattr(loader, "_extract_images", fail_extraction)

    with caplog.at_level(logging.WARNING):
        document = loader.load(str(pdf_path))

    assert "Simple PDF document for RAG" in document.text
    assert document.metadata["images"] == []
    assert "image extraction unavailable" in caplog.text


@pytest.mark.unit
def test_pdf_loader_rejects_non_pdf_input(tmp_path: Path) -> None:
    text_path = tmp_path / "sample.txt"
    text_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="PDF"):
        PdfLoader(images_root=tmp_path / "images").load(str(text_path))


@pytest.mark.unit
def test_base_loader_requires_load_implementation() -> None:
    class IncompleteLoader(BaseLoader):
        pass

    with pytest.raises(TypeError):
        IncompleteLoader()
