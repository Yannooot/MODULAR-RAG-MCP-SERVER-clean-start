"""Registry-backed factory for embedding providers."""

from typing import ClassVar

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactoryError(ValueError):
    """Raised when an embedding provider cannot be registered or created."""


class EmbeddingFactory:
    _providers: ClassVar[dict[str, type[BaseEmbedding]]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseEmbedding]
    ) -> None:
        if not isinstance(provider_class, type) or not issubclass(
            provider_class, BaseEmbedding
        ):
            raise TypeError("Embedding provider must implement BaseEmbedding")
        cls._providers[cls._normalize_name(name)] = provider_class

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        provider_name = cls._normalize_name(settings.embedding.provider)
        provider_class = cls._providers.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise EmbeddingFactoryError(
                f"Unknown embedding provider '{provider_name}'. "
                f"Available providers: {available}"
            )
        return provider_class(settings)

    @classmethod
    def list_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._providers))

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        if not isinstance(name, str) or not name.strip():
            raise EmbeddingFactoryError(
                "Embedding provider name must be a non-empty string"
            )
        return name.strip().lower()
