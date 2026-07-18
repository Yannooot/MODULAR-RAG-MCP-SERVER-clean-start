"""Registry-backed factory for reranking backends."""

from typing import ClassVar

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.llm_reranker import LLMReranker


class RerankerFactoryError(ValueError):
    """Raised when a reranking backend cannot be created."""


class RerankerFactory:
    _providers: ClassVar[dict[str, type[BaseReranker]]] = {
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
        "none": NoneReranker,
    }

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseReranker]
    ) -> None:
        if not isinstance(provider_class, type) or not issubclass(
            provider_class, BaseReranker
        ):
            raise TypeError("Reranker provider must implement BaseReranker")
        cls._providers[cls._normalize_name(name)] = provider_class

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        backend_name = cls._normalize_name(settings.rerank.backend)
        provider_class = cls._providers.get(backend_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise RerankerFactoryError(
                f"Unknown reranker backend '{backend_name}'. "
                f"Available providers: {available}"
            )
        return provider_class(settings)

    @classmethod
    def list_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._providers))

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        if not isinstance(name, str) or not name.strip():
            raise RerankerFactoryError(
                "Reranker backend must be a non-empty string"
            )
        return name.strip().lower()
