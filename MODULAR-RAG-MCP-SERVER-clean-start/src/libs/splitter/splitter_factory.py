"""Registry-backed factory for text splitting strategies."""

from typing import ClassVar

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter


class SplitterFactoryError(ValueError):
    """Raised when a splitter provider cannot be registered or created."""


class SplitterFactory:
    _providers: ClassVar[dict[str, type[BaseSplitter]]] = {
        "recursive": RecursiveSplitter
    }

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseSplitter]
    ) -> None:
        if not isinstance(provider_class, type) or not issubclass(
            provider_class, BaseSplitter
        ):
            raise TypeError("Splitter provider must implement BaseSplitter")
        cls._providers[cls._normalize_name(name)] = provider_class

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        provider_name = cls._normalize_name(settings.splitter.provider)
        provider_class = cls._providers.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise SplitterFactoryError(
                f"Unknown splitter provider '{provider_name}'. "
                f"Available providers: {available}"
            )
        return provider_class(settings)

    @classmethod
    def list_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._providers))

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        if not isinstance(name, str) or not name.strip():
            raise SplitterFactoryError(
                "Splitter provider name must be a non-empty string"
            )
        return name.strip().lower()
