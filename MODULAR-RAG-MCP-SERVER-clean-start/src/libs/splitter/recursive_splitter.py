"""Markdown-aware recursive text splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class RecursiveSplitter(BaseSplitter):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        chunk_size = settings.splitter.chunk_size
        chunk_overlap = settings.splitter.chunk_overlap
        if chunk_size <= 0:
            raise ValueError("splitter.chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "splitter.chunk_overlap must be non-negative and smaller than chunk_size"
            )
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(
        self, text: str, trace: TraceContext | None = None
    ) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return []
        return self._splitter.split_text(text)
