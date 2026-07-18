"""Lightweight deterministic retrieval metrics."""

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, float]:
        golden_id_set = set(golden_ids)
        for rank, document_id in enumerate(retrieved_ids, start=1):
            if document_id in golden_id_set:
                return {"hit_rate": 1.0, "mrr": 1.0 / rank}
        return {"hit_rate": 0.0, "mrr": 0.0}
