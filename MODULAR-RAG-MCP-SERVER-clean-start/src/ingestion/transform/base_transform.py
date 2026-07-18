"""Contract for ingestion chunk transforms."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk


class BaseTransform(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def transform(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        raise NotImplementedError
