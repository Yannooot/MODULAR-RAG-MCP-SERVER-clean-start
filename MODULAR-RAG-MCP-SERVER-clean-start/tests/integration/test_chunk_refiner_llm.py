import os
from dataclasses import replace

import pytest

from core.settings import load_settings
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_LLM_INTEGRATION") != "1",
        reason="set RUN_LLM_INTEGRATION=1 to call the real LLM",
    ),
]


def make_chunk(text: str) -> Chunk:
    return Chunk(
        id="integration-chunk",
        text=text,
        metadata={"source_path": "tests/integration/sample.md"},
        start_offset=0,
        end_offset=len(text),
    )


def test_real_deepseek_refines_noisy_text() -> None:
    settings = load_settings()
    result = ChunkRefiner(settings).transform(
        [make_chunk("页眉：内部资料\n\n向量数据库   用于保存和检索向量。\n\n第 1 页")]
    )[0]

    assert result.metadata["refined_by"] == "llm"
    assert "向量数据库" in result.text
    assert "页眉" not in result.text
    assert "第 1 页" not in result.text


def test_invalid_real_model_falls_back_to_rules() -> None:
    settings = load_settings()
    invalid_llm = replace(settings.llm, model="invalid-model-for-fallback-test")
    invalid_settings = replace(settings, llm=invalid_llm)

    result = ChunkRefiner(invalid_settings).transform(
        [make_chunk("页眉：内部资料\n有效   内容。\n第 1 页")]
    )[0]

    assert result.text == "有效 内容。"
    assert result.metadata["refined_by"] == "rule"
    assert result.metadata["refinement_fallback_reason"]
