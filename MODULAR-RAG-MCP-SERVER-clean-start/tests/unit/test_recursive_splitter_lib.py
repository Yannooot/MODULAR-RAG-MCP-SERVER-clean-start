from dataclasses import replace

import pytest

from core.settings import Settings, load_settings
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def settings_for(chunk_size: int = 120, chunk_overlap: int = 20) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        splitter=replace(
            settings.splitter,
            provider="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )


@pytest.mark.unit
def test_factory_creates_recursive_splitter() -> None:
    settings = settings_for()

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, RecursiveSplitter)
    assert splitter.settings is settings


@pytest.mark.unit
def test_markdown_heading_and_code_block_stay_intact() -> None:
    code_block = '```python\ndef greet():\n    return "hello"\n```'
    markdown = (
        "# Introduction\n\n"
        "A short introduction to the document.\n\n"
        "## Example\n\n"
        f"{code_block}\n\n"
        "## Details\n\n"
        "More details about the example. " * 4
    )
    splitter = SplitterFactory.create(settings_for(chunk_size=100, chunk_overlap=10))

    chunks = splitter.split_text(markdown)

    assert any(chunk.startswith("# Introduction") for chunk in chunks)
    assert any(code_block in chunk for chunk in chunks)
    assert any("## Details" in chunk for chunk in chunks)


@pytest.mark.unit
def test_chunks_respect_configured_size() -> None:
    splitter = SplitterFactory.create(settings_for(chunk_size=40, chunk_overlap=5))

    chunks = splitter.split_text("sentence one. " * 20)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 40 for chunk in chunks)


@pytest.mark.unit
def test_empty_text_returns_no_chunks() -> None:
    splitter = SplitterFactory.create(settings_for())

    assert splitter.split_text("") == []
    assert splitter.split_text("   \n") == []
