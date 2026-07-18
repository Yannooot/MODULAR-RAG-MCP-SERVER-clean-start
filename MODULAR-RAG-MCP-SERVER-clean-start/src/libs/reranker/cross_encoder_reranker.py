"""Cross-encoder candidate reranking."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from numbers import Real
from typing import TYPE_CHECKING, Any

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class CrossEncoderRerankerError(RuntimeError):
    """Signals that callers should use the pre-rerank candidate order."""

    fallback_available = True


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        settings: Settings,
        scorer: Callable[[list[tuple[str, str]]], Iterable[float]] | None = None,
    ) -> None:
        super().__init__(settings)
        if settings.rerank.top_m <= 0:
            raise CrossEncoderRerankerError("rerank.top_m must be greater than zero")
        self._scorer = scorer
        self._model: Any = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        count = min(self.settings.rerank.top_m, len(candidates))
        top_candidates = candidates[:count]
        pairs = [(query, self._candidate_text(candidate)) for candidate in top_candidates]
        try:
            raw_scores = self._score(pairs)
        except Exception as exc:
            raise CrossEncoderRerankerError(
                f"Cross-encoder scoring failed: {exc}"
            ) from exc

        scores = self._validate_scores(raw_scores, count)
        ranked = [
            candidate
            for candidate, _ in sorted(
                zip(top_candidates, scores), key=lambda item: item[1], reverse=True
            )
        ]
        return ranked + candidates[count:]

    @staticmethod
    def _candidate_text(candidate: dict[str, Any]) -> str:
        text = candidate.get("text")
        if not isinstance(text, str):
            raise CrossEncoderRerankerError(
                "Each candidate must contain a string text field"
            )
        return text

    def _score(self, pairs: list[tuple[str, str]]) -> Iterable[float]:
        if self._scorer is not None:
            return self._scorer(pairs)
        if self._model is None:
            self._model = self._load_model()
        return self._model.predict(pairs)

    def _load_model(self) -> Any:
        model_name = self.settings.rerank.model.strip()
        if not model_name:
            raise ValueError("rerank.model must be configured")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for cross_encoder reranking"
            ) from exc
        return CrossEncoder(model_name)

    @staticmethod
    def _validate_scores(raw_scores: Iterable[float], expected: int) -> list[float]:
        try:
            scores = list(raw_scores)
        except TypeError as exc:
            raise CrossEncoderRerankerError("Scorer must return iterable scores") from exc
        if len(scores) != expected or any(
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
            for score in scores
        ):
            raise CrossEncoderRerankerError(
                f"Scorer must return {expected} finite numeric scores"
            )
        return [float(score) for score in scores]
