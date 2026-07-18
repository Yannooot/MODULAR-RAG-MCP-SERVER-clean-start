from dataclasses import replace

import pytest

from core.settings import Settings, load_settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError


class FakeEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, float]:
        return {"fake_metric": 1.0}


@pytest.fixture(autouse=True)
def isolate_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from libs.evaluator.custom_evaluator import CustomEvaluator

    monkeypatch.setattr(EvaluatorFactory, "_providers", {"custom": CustomEvaluator})


def settings_for(backends: list[str]) -> Settings:
    settings = load_settings()
    return replace(settings, evaluation=replace(settings.evaluation, backends=backends))


@pytest.mark.unit
def test_custom_evaluator_calculates_hit_rate_and_mrr() -> None:
    evaluator = EvaluatorFactory.create(settings_for(["custom"]))

    metrics = evaluator.evaluate("query", ["doc-1", "doc-2", "doc-3"], ["doc-2"])

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


@pytest.mark.unit
def test_custom_evaluator_returns_zero_metrics_when_no_document_matches() -> None:
    evaluator = EvaluatorFactory.create(settings_for(["custom"]))

    metrics = evaluator.evaluate("query", ["doc-1"], ["doc-2"])

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


@pytest.mark.unit
def test_factory_routes_to_registered_backend() -> None:
    EvaluatorFactory.register_provider("fake", FakeEvaluator)
    settings = settings_for(["fake"])

    evaluator = EvaluatorFactory.create(settings)

    assert isinstance(evaluator, FakeEvaluator)
    assert evaluator.settings is settings
    assert evaluator.evaluate("query", [], []) == {"fake_metric": 1.0}


@pytest.mark.unit
def test_unknown_backend_has_readable_error() -> None:
    with pytest.raises(EvaluatorFactoryError, match="unknown-backend"):
        EvaluatorFactory.create(settings_for(["unknown-backend"]))


@pytest.mark.unit
def test_registered_provider_must_implement_base_evaluator() -> None:
    with pytest.raises(TypeError, match="BaseEvaluator"):
        EvaluatorFactory.register_provider("invalid", object)


@pytest.mark.unit
def test_base_evaluator_requires_evaluate_implementation() -> None:
    class IncompleteEvaluator(BaseEvaluator):
        pass

    with pytest.raises(TypeError):
        IncompleteEvaluator(settings_for(["custom"]))
