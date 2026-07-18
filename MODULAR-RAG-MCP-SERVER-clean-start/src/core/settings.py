"""Application configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class SettingsError(ValueError):
    """Raised when application configuration cannot be loaded or validated."""


@dataclass(frozen=True)
class ProviderSettings:
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class VisionLLMSettings(ProviderSettings):
    azure_endpoint: str | None = None
    api_version: str = "2024-02-15-preview"
    deployment_name: str | None = None
    max_image_size: int = 2048
    max_tokens: int = 1000


@dataclass(frozen=True)
class SplitterSettings:
    provider: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass(frozen=True)
class VectorStoreSettings:
    backend: str | None = None
    persist_path: str | None = None
    collection_name: str = "chunks"


@dataclass(frozen=True)
class RetrievalSettings:
    sparse_backend: str | None = None
    fusion_algorithm: str | None = None
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10


@dataclass(frozen=True)
class RerankSettings:
    backend: str | None = None
    model: str = ""
    top_m: int = 30


@dataclass(frozen=True)
class EvaluationSettings:
    backends: list[str] = field(default_factory=list)
    golden_test_set: str | None = None


@dataclass(frozen=True)
class ObservabilitySettings:
    enabled: bool | None = None
    log_file: str | None = None


@dataclass(frozen=True)
class ChunkRefinerSettings:
    use_llm: bool = False


@dataclass(frozen=True)
class IngestionSettings:
    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)


@dataclass(frozen=True)
class Settings:
    llm: ProviderSettings
    vision_llm: VisionLLMSettings
    embedding: ProviderSettings
    splitter: SplitterSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)


def validate_settings(settings: Settings) -> None:
    """Validate required fields and report their full configuration paths."""
    required_fields = (
        ("llm.provider", settings.llm.provider),
        ("llm.model", settings.llm.model),
        ("embedding.provider", settings.embedding.provider),
        ("embedding.model", settings.embedding.model),
        ("vector_store.backend", settings.vector_store.backend),
        ("vector_store.persist_path", settings.vector_store.persist_path),
        ("retrieval.sparse_backend", settings.retrieval.sparse_backend),
        ("retrieval.fusion_algorithm", settings.retrieval.fusion_algorithm),
        ("rerank.backend", settings.rerank.backend),
        ("evaluation.backends", settings.evaluation.backends),
        ("observability.enabled", settings.observability.enabled),
        ("observability.log_file", settings.observability.log_file),
    )
    for field_path, value in required_fields:
        if value is None or value == "" or value == []:
            raise SettingsError(f"Missing required field: {field_path}")


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load YAML configuration into typed settings and validate it."""
    config_path = Path(path)
    if not config_path.is_file():
        raise SettingsError(f"Configuration file not found: {config_path}")

    try:
        load_dotenv(dotenv_path=Path(".env"), override=False)
        content = os.path.expandvars(config_path.read_text(encoding="utf-8"))
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML in configuration file: {exc}") from exc

    if not isinstance(raw, dict):
        raise SettingsError("Configuration root must be a mapping")

    settings = Settings(
        llm=_provider_settings(raw.get("llm")),
        vision_llm=_vision_llm_settings(raw.get("vision_llm")),
        embedding=_provider_settings(raw.get("embedding")),
        splitter=SplitterSettings(**_section(raw, "splitter")),
        vector_store=VectorStoreSettings(**_section(raw, "vector_store")),
        retrieval=RetrievalSettings(**_section(raw, "retrieval")),
        rerank=RerankSettings(**_section(raw, "rerank")),
        evaluation=EvaluationSettings(**_section(raw, "evaluation")),
        observability=ObservabilitySettings(**_section(raw, "observability")),
        ingestion=_ingestion_settings(raw.get("ingestion")),
    )
    validate_settings(settings)
    return settings


def _provider_settings(value: Any) -> ProviderSettings:
    section = value if isinstance(value, dict) else {}
    return ProviderSettings(
        provider=section.get("provider"),
        model=section.get("model"),
        api_key=section.get("api_key"),
        base_url=section.get("base_url"),
    )


def _vision_llm_settings(value: Any) -> VisionLLMSettings:
    section = value if isinstance(value, dict) else {}
    return VisionLLMSettings(
        provider=section.get("provider"),
        model=section.get("model"),
        api_key=section.get("api_key"),
        base_url=section.get("base_url"),
        azure_endpoint=section.get("azure_endpoint"),
        api_version=section.get("api_version", "2024-02-15-preview"),
        deployment_name=section.get("deployment_name"),
        max_image_size=section.get("max_image_size", 2048),
        max_tokens=section.get("max_tokens", 1000),
    )


def _ingestion_settings(value: Any) -> IngestionSettings:
    section = value if isinstance(value, dict) else {}
    chunk_refiner = section.get("chunk_refiner")
    chunk_refiner_section = chunk_refiner if isinstance(chunk_refiner, dict) else {}
    return IngestionSettings(
        chunk_refiner=ChunkRefinerSettings(**chunk_refiner_section)
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    return value if isinstance(value, dict) else {}
