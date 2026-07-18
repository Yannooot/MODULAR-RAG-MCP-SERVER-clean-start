"""Common interface for evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.settings import Settings


class BaseEvaluator(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, float]:
        """Evaluate retrieval results against the expected document IDs."""
        raise NotImplementedError
