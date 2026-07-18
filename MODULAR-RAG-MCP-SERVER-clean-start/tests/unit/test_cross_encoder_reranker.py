from dataclasses import replace
from typing import Any

import pytest

from core.settings import Settings, load_settings
from libs.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from libs.reranker.reranker_factory import RerankerFactory


class StubScorer:
    def __init__(
        self,
        scores: list[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores or []
        self.error = error
        self.pairs: list[tuple[str, str]] | None = None

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        if self.error is not None:
            raise self.error
        return self.scores


def settings_for(top_m: int = 30) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        rerank=replace(
            settings.rerank,
            backend="cross_encoder",
            model="test-cross-encoder",
            top_m=top_m,
        ),
    )


@pytest.mark.unit
def test_factory_creates_cross_encoder_without_loading_model() -> None:
    reranker = RerankerFactory.create(settings_for())

    assert isinstance(reranker, CrossEncoderReranker)


@pytest.mark.unit
def test_scores_top_m_and_keeps_remaining_candidate_order() -> None:
    scorer = StubScorer([0.2, 0.9])
    candidates = [
        {"id": "first", "text": "First passage"},
        {"id": "second", "text": "Second passage"},
        {"id": "third", "text": "Third passage"},
    ]
    reranker = CrossEncoderReranker(settings_for(top_m=2), scorer=scorer)

    ranked = reranker.rerank("query", candidates)

    assert ranked == [candidates[1], candidates[0], candidates[2]]
    assert scorer.pairs == [
        ("query", "First passage"),
        ("query", "Second passage"),
    ]


@pytest.mark.unit
def test_equal_scores_keep_original_order() -> None:
    candidates = [{"id": "first", "text": "One"}, {"id": "second", "text": "Two"}]
    reranker = CrossEncoderReranker(
        settings_for(), scorer=StubScorer([0.5, 0.5])
    )

    assert reranker.rerank("query", candidates) == candidates


@pytest.mark.unit
@pytest.mark.parametrize(
    "scores",
    [[], [0.1], [0.1, "invalid"]],
)
def test_invalid_scores_raise_readable_fallback_error(scores: list[Any]) -> None:
    reranker = CrossEncoderReranker(settings_for(), scorer=StubScorer(scores))
    candidates = [{"id": "first", "text": "One"}, {"id": "second", "text": "Two"}]

    with pytest.raises(CrossEncoderRerankerError, match="scores") as exc_info:
        reranker.rerank("query", candidates)

    assert exc_info.value.fallback_available is True


@pytest.mark.unit
def test_scorer_timeout_raises_fallback_error() -> None:
    reranker = CrossEncoderReranker(
        settings_for(), scorer=StubScorer(error=TimeoutError("timed out"))
    )

    with pytest.raises(CrossEncoderRerankerError, match="timed out") as exc_info:
        reranker.rerank("query", [{"id": "first", "text": "One"}])

    assert exc_info.value.fallback_available is True
