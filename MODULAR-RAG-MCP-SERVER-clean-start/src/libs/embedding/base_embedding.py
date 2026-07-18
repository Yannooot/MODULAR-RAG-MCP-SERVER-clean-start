"""Common interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        """Return one vector for each input text."""
        raise NotImplementedError
