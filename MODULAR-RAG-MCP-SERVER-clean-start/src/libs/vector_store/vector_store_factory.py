"""Registry-backed factory for vector storage backends."""

from typing import ClassVar

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore


class VectorStoreFactoryError(ValueError):
    """Raised when a vector storage backend cannot be created."""


class VectorStoreFactory:
    _providers: ClassVar[dict[str, type[BaseVectorStore]]] = {
        "chroma": ChromaStore
    }

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseVectorStore]
    ) -> None:
        if not isinstance(provider_class, type) or not issubclass(
            provider_class, BaseVectorStore
        ):
            raise TypeError("Vector store provider must implement BaseVectorStore")
        cls._providers[cls._normalize_name(name)] = provider_class

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        provider_name = cls._normalize_name(settings.vector_store.backend)
        provider_class = cls._providers.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise VectorStoreFactoryError(
                f"Unknown vector store backend '{provider_name}'. "
                f"Available providers: {available}"
            )
        return provider_class(settings)

    @classmethod
    def list_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._providers))

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        if not isinstance(name, str) or not name.strip():
            raise VectorStoreFactoryError(
                "Vector store backend must be a non-empty string"
            )
        return name.strip().lower()
