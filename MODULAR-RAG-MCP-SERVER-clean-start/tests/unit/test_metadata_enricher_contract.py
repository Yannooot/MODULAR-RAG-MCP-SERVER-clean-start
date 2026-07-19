import json
import os
from dataclasses import replace
from typing import Any

import pytest

from core.settings import (
    IngestionSettings,
    MetadataEnricherSettings,
    Settings,
    load_settings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher


class FakeLLM:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.messages: Any = None

    def chat(self, messages: Any) -> str:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


def settings(use_llm: bool = False) -> Settings:
    app_settings = load_settings()
    return replace(
        app_settings,
        ingestion=replace(
            app_settings.ingestion,
            metadata_enricher=MetadataEnricherSettings(use_llm=use_llm),
        ),
    )


def chunk(
    text: str,
    identifier: str = "chunk-1",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        metadata=metadata or {"source_path": "docs/vector.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_rule_mode_always_adds_non_empty_metadata() -> None:
    result = MetadataEnricher(settings()).transform(
        [chunk("# 向量检索\n\n向量数据库支持语义相似度搜索。")]
    )[0]

    assert result.metadata["title"] == "向量检索"
    assert "向量数据库" in result.metadata["summary"]
    assert result.metadata["tags"]
    assert result.metadata["metadata_enriched_by"] == "rule"


@pytest.mark.unit
def test_rule_mode_preserves_existing_title() -> None:
    result = MetadataEnricher(settings()).transform(
        [
            chunk(
                "正文第一段。正文第二段。",
                metadata={"source_path": "docs/vector.md", "title": "已有标题"},
            )
        ]
    )[0]

    assert result.metadata["title"] == "已有标题"
    assert result.metadata["summary"] == "正文第一段。正文第二段。"


@pytest.mark.unit
def test_rule_mode_handles_empty_text() -> None:
    result = MetadataEnricher(settings()).transform([chunk("")])[0]

    assert all(result.metadata[field] for field in ("title", "summary", "tags"))


@pytest.mark.unit
def test_llm_mode_parses_structured_metadata() -> None:
    response = json.dumps(
        {
            "title": "向量数据库检索",
            "summary": "介绍向量数据库如何执行语义检索。",
            "tags": ["向量数据库", "语义检索", "RAG"],
        },
        ensure_ascii=False,
    )
    llm = FakeLLM(response)
    source = chunk("向量数据库通过向量距离完成语义检索。")

    result = MetadataEnricher(settings(use_llm=True), llm=llm).transform([source])[0]

    assert result.metadata["title"] == "向量数据库检索"
    assert result.metadata["summary"] == "介绍向量数据库如何执行语义检索。"
    assert result.metadata["tags"] == ["向量数据库", "语义检索", "RAG"]
    assert result.metadata["metadata_enriched_by"] == "llm"
    assert source.text in llm.messages[0]["content"]
    assert source.metadata == {"source_path": "docs/vector.md"}


@pytest.mark.unit
def test_llm_mode_accepts_json_code_fence() -> None:
    llm = FakeLLM(
        '```json\n{"title":"标题","summary":"摘要","tags":["标签"]}\n```'
    )

    result = MetadataEnricher(settings(use_llm=True), llm=llm).transform(
        [chunk("正文")]
    )[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert result.metadata["tags"] == ["标签"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"title":"","summary":"摘要","tags":["标签"]}',
        '{"title":"标题","summary":"摘要","tags":[]}',
    ],
)
def test_invalid_llm_metadata_falls_back_to_rules(response: str) -> None:
    result = MetadataEnricher(
        settings(use_llm=True), llm=FakeLLM(response)
    ).transform([chunk("# 规则标题\n\n规则摘要内容。")])[0]

    assert result.metadata["title"] == "规则标题"
    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_enrichment_fallback_reason"]


@pytest.mark.unit
def test_llm_exception_falls_back_without_raising() -> None:
    result = MetadataEnricher(
        settings(use_llm=True), llm=FakeLLM(error=RuntimeError("service offline"))
    ).transform([chunk("可用的规则内容。")])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert "service offline" in result.metadata["metadata_enrichment_fallback_reason"]


@pytest.mark.unit
def test_disabled_llm_is_not_called() -> None:
    llm = FakeLLM(error=AssertionError("must not be called"))

    result = MetadataEnricher(settings(), llm=llm).transform([chunk("规则内容。")])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"


@pytest.mark.unit
def test_one_chunk_failure_does_not_stop_following_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enricher = MetadataEnricher(settings())
    original = enricher._rule_metadata

    def build_metadata(source: Chunk) -> dict[str, Any]:
        if source.id == "bad":
            raise ValueError("broken chunk")
        return original(source)

    monkeypatch.setattr(enricher, "_rule_metadata", build_metadata)
    result = enricher.transform([chunk("bad", "bad"), chunk("good", "good")])

    assert result[0].text == "bad"
    assert result[0].metadata["metadata_enriched_by"] == "original"
    assert "broken chunk" in result[0].metadata["metadata_enrichment_fallback_reason"]
    assert result[1].metadata["metadata_enriched_by"] == "rule"


@pytest.mark.unit
def test_transform_records_trace_summary() -> None:
    trace = TraceContext()

    MetadataEnricher(settings()).transform([chunk("content")], trace)

    assert trace.stages[-1] == {
        "name": "metadata_enricher",
        "details": {"chunk_count": 1, "llm_enabled": False},
    }


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION") != "1",
    reason="set RUN_LLM_INTEGRATION=1 to call the real LLM",
)
def test_real_deepseek_generates_semantic_metadata() -> None:
    result = MetadataEnricher(load_settings()).transform(
        [chunk("RAG 系统先检索知识库中的相关文档，再让大语言模型根据证据生成答案。")]
    )[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert result.metadata["title"]
    assert "RAG" in result.metadata["summary"]
    assert len(result.metadata["tags"]) >= 2
