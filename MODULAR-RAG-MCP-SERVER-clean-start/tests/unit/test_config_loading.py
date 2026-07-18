from dataclasses import is_dataclass
from pathlib import Path

import pytest

from core.settings import SettingsError, load_settings


VALID_CONFIG = """
llm:
  provider: deepseek
  model: deepseek-chat
embedding:
  provider: ollama
  model: nomic-embed-text
vector_store:
  backend: chroma
  persist_path: ./data/db/chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 20
  top_k_sparse: 20
  top_k_final: 10
rerank:
  backend: none
  model: ""
  top_m: 30
evaluation:
  backends: [custom]
  golden_test_set: ./tests/fixtures/golden_test_set.json
observability:
  enabled: true
  log_file: ./logs/traces.jsonl
"""


def write_config(tmp_path: Path, content: str = VALID_CONFIG) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
def test_load_settings_returns_structured_settings(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path))

    assert is_dataclass(settings)
    assert settings.llm.provider == "deepseek"
    assert settings.embedding.model == "nomic-embed-text"
    assert settings.vector_store.backend == "chroma"


@pytest.mark.unit
def test_default_config_loads() -> None:
    settings = load_settings()

    assert settings.llm.provider == "deepseek"


@pytest.mark.unit
def test_environment_variable_placeholder_is_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    config = VALID_CONFIG.replace(
        "  model: deepseek-chat\n",
        "  model: deepseek-chat\n  api_key: ${DEEPSEEK_API_KEY}\n",
    )

    settings = load_settings(write_config(tmp_path, config))

    assert settings.llm.api_key == "test-secret"


@pytest.mark.unit
def test_missing_required_field_reports_field_path(tmp_path: Path) -> None:
    invalid_config = VALID_CONFIG.replace("  provider: ollama\n", "")

    with pytest.raises(SettingsError, match=r"embedding\.provider"):
        load_settings(write_config(tmp_path, invalid_config))


@pytest.mark.unit
def test_invalid_yaml_has_readable_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="Invalid YAML"):
        load_settings(write_config(tmp_path, "llm: [invalid"))


@pytest.mark.unit
def test_missing_config_file_has_readable_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(SettingsError, match="Configuration file not found"):
        load_settings(missing_path)
