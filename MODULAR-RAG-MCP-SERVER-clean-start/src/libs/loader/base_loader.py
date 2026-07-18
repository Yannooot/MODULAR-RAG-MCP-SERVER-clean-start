"""Common interface for document loaders."""

from abc import ABC, abstractmethod

from core.types import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        """Load a source file into the shared Document contract."""
        raise NotImplementedError
