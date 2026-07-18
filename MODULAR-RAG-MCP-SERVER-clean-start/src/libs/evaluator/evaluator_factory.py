"""Registry-backed factory for evaluation backends."""

from typing import ClassVar

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactoryError(ValueError):
    """Raised when an evaluation backend cannot be created."""


class EvaluatorFactory:
    _providers: ClassVar[dict[str, type[BaseEvaluator]]] = {"custom": CustomEvaluator}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseEvaluator]
    ) -> None:
        if not isinstance(provider_class, type) or not issubclass(
            provider_class, BaseEvaluator
        ):
            raise TypeError("Evaluator provider must implement BaseEvaluator")
        cls._providers[cls._normalize_name(name)] = provider_class

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        backend_name = cls._configured_backend(settings)
        provider_class = cls._providers.get(backend_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise EvaluatorFactoryError(
                f"Unknown evaluator backend '{backend_name}'. "
                f"Available providers: {available}"
            )
        return provider_class(settings)

    @classmethod
    def list_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._providers))

    @classmethod
    def _configured_backend(cls, settings: Settings) -> str:
        if not settings.evaluation.backends:
            raise EvaluatorFactoryError("Evaluation backends must contain a backend name")
        return cls._normalize_name(settings.evaluation.backends[0])

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        if not isinstance(name, str) or not name.strip():
            raise EvaluatorFactoryError(
                "Evaluator backend must be a non-empty string"
            )
        return name.strip().lower()
