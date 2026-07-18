"""Common interface for reranking backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseReranker(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates ordered by relevance to the query."""
        raise NotImplementedError


class NoneReranker(BaseReranker):
    """Fallback backend that leaves the existing ranking untouched."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        return candidates
