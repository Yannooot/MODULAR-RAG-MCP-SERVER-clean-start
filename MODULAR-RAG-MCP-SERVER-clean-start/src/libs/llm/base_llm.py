"""Common interface for text LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from core.settings import Settings


class BaseLLM(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the provider's text response for a chat conversation."""
        raise NotImplementedError
