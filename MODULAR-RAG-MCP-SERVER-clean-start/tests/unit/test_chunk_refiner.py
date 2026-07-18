import json
from abc import ABC
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from core.settings import ChunkRefinerSettings, IngestionSettings, Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner


class FakeLLM:
    def __init__(self, response: str = "精炼后的文本", error: Exception | None = None):
        self.response = response
        self.error = error
        self.messages: Any = None

    def chat(self, messages: Any) -> str:
        self.messages = messages
        if self.error:
            raise self.error
        return self.response


def make_settings(use_llm: bool = False) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        ingestion=IngestionSettings(
            chunk_refiner=ChunkRefinerSettings(use_llm=use_llm)
        ),
    )


def make_chunk(text: str, identifier: str = "chunk-1") -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        metadata={"source_path": "docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


@pytest.fixture(scope="module")
def noisy_cases() -> list[dict[str, str]]:
    path = Path(__file__).parents[1] / "fixtures" / "noisy_chunks.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_base_transform_is_abstract() -> None:
    assert issubclass(BaseTransform, ABC)
    with pytest.raises(TypeError):
        BaseTransform(make_settings())  # type: ignore[abstract]


@pytest.mark.unit
def test_trace_context_records_stage_data() -> None:
    trace = TraceContext()
    trace.record_stage("transform", {"chunk_count": 2})

    assert trace.trace_id
    assert trace.stages == [
        {"name": "transform", "details": {"chunk_count": 2}}
    ]


@pytest.mark.unit
def test_default_config_enables_llm_refinement() -> None:
    assert load_settings().ingestion.chunk_refiner.use_llm is True


@pytest.mark.unit
@pytest.mark.parametrize("case_index", range(8))
def test_rule_refinement_handles_fixture_cases(
    noisy_cases: list[dict[str, str]], case_index: int
) -> None:
    case = noisy_cases[case_index]

    result = ChunkRefiner(make_settings())._rule_based_refine(case["input"])

    assert result == case["expected"], case["name"]


@pytest.mark.unit
def test_transform_uses_llm_and_preserves_chunk_contract() -> None:
    llm = FakeLLM("更清晰的内容。")
    source = make_chunk("  原始  内容。 ")

    result = ChunkRefiner(make_settings(use_llm=True), llm=llm).transform([source])

    assert result[0].text == "更清晰的内容。"
    assert result[0].metadata["refined_by"] == "llm"
    assert result[0].id == source.id
    assert result[0].start_offset == source.start_offset
    assert "原始 内容。" in llm.messages[0]["content"]
    assert source.text == "  原始  内容。 "
    assert "refined_by" not in source.metadata


@pytest.mark.unit
@pytest.mark.parametrize("response", ["", "   "])
def test_empty_llm_response_falls_back_to_rules(response: str) -> None:
    result = ChunkRefiner(
        make_settings(use_llm=True), llm=FakeLLM(response)
    ).transform([make_chunk(" 内容   保留。 ")])

    assert result[0].text == "内容 保留。"
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].metadata["refinement_fallback_reason"] == "empty LLM response"


@pytest.mark.unit
def test_llm_error_falls_back_to_rules() -> None:
    result = ChunkRefiner(
        make_settings(use_llm=True), llm=FakeLLM(error=RuntimeError("offline"))
    ).transform([make_chunk(" 内容   保留。 ")])

    assert result[0].text == "内容 保留。"
    assert result[0].metadata["refined_by"] == "rule"
    assert "offline" in result[0].metadata["refinement_fallback_reason"]


@pytest.mark.unit
def test_disabled_llm_is_not_called() -> None:
    llm = FakeLLM(error=AssertionError("must not be called"))

    result = ChunkRefiner(make_settings(use_llm=False), llm=llm).transform(
        [make_chunk(" 规则   清洗 ")]
    )

    assert result[0].text == "规则 清洗"
    assert result[0].metadata == {
        "source_path": "docs/sample.md",
        "refined_by": "rule",
    }


@pytest.mark.unit
def test_prompt_file_and_missing_file_fallback(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("处理：{text}", encoding="utf-8")

    loaded = ChunkRefiner(make_settings(), prompt_path=prompt)
    fallback = ChunkRefiner(make_settings(), prompt_path=tmp_path / "missing.txt")

    assert loaded.prompt_template == "处理：{text}"
    assert "{text}" in fallback.prompt_template


@pytest.mark.unit
def test_one_chunk_failure_does_not_stop_other_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refiner = ChunkRefiner(make_settings())
    original = refiner._rule_based_refine

    def refine(text: str) -> str:
        if text == "bad":
            raise ValueError("broken chunk")
        return original(text)

    monkeypatch.setattr(refiner, "_rule_based_refine", refine)
    result = refiner.transform([make_chunk("bad", "bad"), make_chunk(" good ", "good")])

    assert result[0].text == "bad"
    assert result[0].metadata["refined_by"] == "original"
    assert "broken chunk" in result[0].metadata["refinement_fallback_reason"]
    assert result[1].text == "good"


@pytest.mark.unit
def test_transform_records_trace_summary() -> None:
    trace = TraceContext()

    ChunkRefiner(make_settings()).transform([make_chunk(" content ")], trace)

    assert trace.stages[-1] == {
        "name": "chunk_refiner",
        "details": {"chunk_count": 1, "llm_enabled": False},
    }
