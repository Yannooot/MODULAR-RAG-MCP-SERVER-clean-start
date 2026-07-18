"""Common interface for vision LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.settings import Settings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


ChatResponse = str


class BaseVisionLLM(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def preprocess_image(self, image_path: str | bytes) -> str | bytes:
        """Prepare an image before sending it to a provider."""
        return image_path

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """Return a response for text combined with an image path or bytes."""
        raise NotImplementedError
