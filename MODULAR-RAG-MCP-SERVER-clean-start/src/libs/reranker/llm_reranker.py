"""LLM-backed candidate reranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class LLMRerankerError(RuntimeError):
    """Signals that callers should use the pre-rerank candidate order."""

    fallback_available = True


class LLMReranker(BaseReranker):
    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_text: str | None = None,
    ) -> None:
        super().__init__(settings)
        try:
            self._llm = llm or LLMFactory.create(settings)
            self._prompt_text = prompt_text or self._load_prompt()
        except Exception as exc:
            raise LLMRerankerError(f"LLM reranker setup failed: {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        candidates_by_id = self._index_candidates(candidates)
        prompt = self._build_prompt(query, candidates)
        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            raise LLMRerankerError(f"LLM reranking failed: {exc}") from exc

        ranked_ids = self._parse_ranked_ids(response, set(candidates_by_id))
        return [candidates_by_id[candidate_id] for candidate_id in ranked_ids]

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).resolve().parents[3] / "config" / "prompts" / "rerank.txt"
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Rerank prompt is empty: {prompt_path}")
        return prompt

    @staticmethod
    def _index_candidates(
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(candidates):
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise LLMRerankerError(
                    f"Candidate at index {index} must have a non-empty string id"
                )
            if candidate_id in indexed:
                raise LLMRerankerError(f"Duplicate candidate id: {candidate_id}")
            indexed[candidate_id] = candidate
        return indexed

    def _build_prompt(self, query: str, candidates: list[dict[str, Any]]) -> str:
        candidate_payload = [
            {"id": candidate["id"], "text": candidate.get("text", "")}
            for candidate in candidates
        ]
        return (
            f"{self._prompt_text}\n\n"
            "Return only JSON with this schema: "
            '{"ranked_ids": ["candidate-id"]}. '
            "Include every candidate id exactly once.\n\n"
            f"Query:\n{query}\n\n"
            f"Candidates:\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _parse_ranked_ids(response: str, expected_ids: set[str]) -> list[str]:
        try:
            payload = json.loads(response)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMRerankerError(
                "LLM reranker response must be valid JSON"
            ) from exc

        if not isinstance(payload, dict) or set(payload) != {"ranked_ids"}:
            raise LLMRerankerError(
                "LLM reranker response must contain only ranked_ids"
            )
        ranked_ids = payload["ranked_ids"]
        if (
            not isinstance(ranked_ids, list)
            or any(not isinstance(candidate_id, str) for candidate_id in ranked_ids)
            or len(ranked_ids) != len(expected_ids)
            or set(ranked_ids) != expected_ids
        ):
            raise LLMRerankerError(
                "ranked_ids must contain every candidate IDs exactly once"
            )
        return ranked_ids
