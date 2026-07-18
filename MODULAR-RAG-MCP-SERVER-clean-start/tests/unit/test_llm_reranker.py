from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.llm.llm_factory import LLMFactory
from libs.reranker.llm_reranker import LLMReranker, LLMRerankerError
from libs.reranker.reranker_factory import RerankerFactory


class StubLLM:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.messages: Any = None

    def chat(self, messages: Any) -> str:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


def settings_for(backend: str = "llm") -> Settings:
    settings = load_settings()
    return replace(settings, rerank=replace(settings.rerank, backend=backend))


@pytest.mark.unit
def test_factory_creates_llm_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = StubLLM('{"ranked_ids": []}')
    monkeypatch.setattr(LLMFactory, "create", lambda settings: llm)

    reranker = RerankerFactory.create(settings_for())

    assert isinstance(reranker, LLMReranker)


@pytest.mark.unit
def test_rerank_uses_prompt_and_orders_original_candidates() -> None:
    llm = StubLLM('{"ranked_ids": ["second", "first"]}')
    candidates = [
        {"id": "first", "text": "First passage", "score": 0.9},
        {"id": "second", "text": "Second passage", "score": 0.5},
    ]
    reranker = LLMReranker(settings_for(), llm=llm, prompt_text="CUSTOM PROMPT")

    ranked = reranker.rerank("Which passage is relevant?", candidates)

    assert ranked == [candidates[1], candidates[0]]
    prompt = llm.messages[0]["content"]
    assert "CUSTOM PROMPT" in prompt
    assert "Which passage is relevant?" in prompt
    assert '"id": "first"' in prompt


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not json", "valid JSON"),
        ('{"ids": ["first"]}', "ranked_ids"),
        ('{"ranked_ids": ["first", "unknown"]}', "candidate IDs"),
        ('{"ranked_ids": ["first", "first"]}', "candidate IDs"),
    ],
)
def test_invalid_llm_output_raises_readable_fallback_error(
    response: str, message: str
) -> None:
    reranker = LLMReranker(
        settings_for(), llm=StubLLM(response), prompt_text="Rank candidates"
    )
    candidates = [{"id": "first", "text": "One"}, {"id": "second", "text": "Two"}]

    with pytest.raises(LLMRerankerError, match=message) as exc_info:
        reranker.rerank("query", candidates)

    assert exc_info.value.fallback_available is True


@pytest.mark.unit
def test_llm_failure_raises_fallback_error() -> None:
    reranker = LLMReranker(
        settings_for(),
        llm=StubLLM(error=TimeoutError("timed out")),
        prompt_text="Rank candidates",
    )

    with pytest.raises(LLMRerankerError, match="timed out") as exc_info:
        reranker.rerank("query", [{"id": "first", "text": "One"}])

    assert exc_info.value.fallback_available is True
