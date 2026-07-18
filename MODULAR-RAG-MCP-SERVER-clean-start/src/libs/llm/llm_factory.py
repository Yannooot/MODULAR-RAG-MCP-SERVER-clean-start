"""Registry-backed factory for text LLM providers."""

from typing import ClassVar

from core.settings import Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import OpenAILLM


class LLMFactoryError(ValueError):
    """Raised when an LLM provider cannot be registered or created."""


class LLMFactory:
    _providers: ClassVar[dict[str, type[BaseLLM]]] = {
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        if not isinstance(provider_class, type) or not issubclass(provider_class, BaseLLM):
            raise TypeError("LLM provider must implement BaseLLM")
        cls._providers[cls._normalize_name(name)] = provider_class

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        provider_name = cls._normalize_name(settings.llm.provider)
        provider_class = cls._providers.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise LLMFactoryError(
                f"Unknown LLM provider '{provider_name}'. Available providers: {available}"
            )
        return provider_class(settings)

    @classmethod
    def list_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._providers))

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        if not isinstance(name, str) or not name.strip():
            raise LLMFactoryError("LLM provider name must be a non-empty string")
        return name.strip().lower()
