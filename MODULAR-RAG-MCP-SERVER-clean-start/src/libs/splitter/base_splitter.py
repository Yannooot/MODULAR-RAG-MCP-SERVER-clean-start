"""Common interface for text splitting strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def split_text(
        self, text: str, trace: TraceContext | None = None
    ) -> list[str]:
        """Split text into chunks while preserving their original order."""
        raise NotImplementedError
