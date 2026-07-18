"""Common contract for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseVectorStore(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def upsert(
        self, records: list[dict[str, Any]], trace: TraceContext | None = None
    ) -> None:
        """Insert or update vector records."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return records containing id, text, metadata, and score."""
        raise NotImplementedError
