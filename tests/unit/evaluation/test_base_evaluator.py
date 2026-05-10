"""Tests for BaseEvaluator abstract class and template method pattern."""

import pytest
from unittest.mock import Mock, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from app.core.enums import Stages
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.core.experiments import ExperimentResult
from app.core.context.context import Context, StageResult, ProjectConfig


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation for testing BaseEvaluator."""

    def _extract_stage_specific_data(self, sorted_results, best_experiment):
        return {"test_data": "test_value", "best_name": best_experiment.name}

    def _update_context(self, sorted_results, best_experiment, stage_specific_data):
        stage_result = StageResult(
            name=self.stage,
            best_experiment=best_experiment,
            metadata=stage_specific_data,
        )
        self.context.update_stage_context(self.stage, stage_result)


class TestBaseEvaluator:
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_accuracy"
        config.scoring = ["accuracy"]
        return config

    @pytest.fixture
    def mock_context(self, mock_config):
        context = Mock(spec=Context)
        context.config = mock_config
        context.stage_results = {}
        context.update_stage_context = MagicMock()
        return context

    @pytest.fixture
    def evaluator(self, mock_context):
        return ConcreteEvaluator(stage=Stages.MODEL_SELECTION, context=mock_context)

    @pytest.fixture
    def sample_results(self):
        results = []
        for i, acc in enumerate([0.85, 0.95, 0.90]):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([("classifier", LogisticRegression())]),
                metrics={"test_accuracy": acc},
                config={"model": f"Model_{i % 2}", "param": f"value_{i}"},
            )
            results.append(result)
        return results

    def test_evaluator_initialization(self, mock_context):
        evaluator = ConcreteEvaluator(
            stage=Stages.FEATURE_SELECTION, context=mock_context
        )
        assert evaluator.stage == Stages.FEATURE_SELECTION
        assert evaluator.context == mock_context
        assert evaluator.config == mock_context.config
        assert evaluator.priority_metric == "test_accuracy"

    def test_evaluate_with_empty_results(self, evaluator):
        evaluator.evaluate([])

    def test_evaluate_calls_template_methods(self, evaluator, sample_results):
        evaluator._sort_results = MagicMock(return_value=sample_results)
        evaluator._extract_stage_specific_data = MagicMock(
            return_value={"test": "data"}
        )
        evaluator._update_context = MagicMock()

        evaluator.evaluate(sample_results)

        evaluator._sort_results.assert_called_once()
        evaluator._extract_stage_specific_data.assert_called_once()
        evaluator._update_context.assert_called_once()

    def test_evaluate_orders_steps_correctly(self, evaluator, sample_results):
        call_order = []

        evaluator._sort_results = Mock(
            side_effect=lambda *a, **k: (call_order.append("sort"), sample_results)[1]
        )
        evaluator._extract_stage_specific_data = Mock(
            side_effect=lambda *a, **k: (
                call_order.append("extract"),
                {"data": "value"},
            )[1]
        )
        evaluator._update_context = Mock(
            side_effect=lambda *a, **k: call_order.append("update")
        )

        evaluator.evaluate(sample_results)

        assert call_order == ["sort", "extract", "update"]

    def test_sort_results_by_accuracy_max_mode(self, evaluator, sample_results):
        sorted_results = evaluator._sort_results(sample_results)

        assert sorted_results[0].metrics["test_accuracy"] == 0.95
        assert sorted_results[1].metrics["test_accuracy"] == 0.90
        assert sorted_results[2].metrics["test_accuracy"] == 0.85

    def test_sort_results_with_list_scoring(self, evaluator, sample_results):
        evaluator.config.scoring = ["accuracy", "precision"]
        sorted_results = evaluator._sort_results(sample_results)
        assert sorted_results[0].metrics["test_accuracy"] == 0.95

    def test_get_model_family_from_config(self, evaluator):
        result = ExperimentResult(
            name="test", pipeline=Pipeline([]), config={"model": "RandomForest"}
        )
        family = evaluator._get_model_family(result)
        assert family == "RandomForest"

    def test_get_model_family_unknown_model(self, evaluator):
        result = ExperimentResult(name="test", pipeline=Pipeline([]), config={})
        family = evaluator._get_model_family(result)
        assert family == "unknown"

    def test_extract_top_k_by_family_single_per_family(self, evaluator):
        results = []
        models = ["RF", "RF", "SVM", "SVM", "LR", "LR"]
        metrics = [0.95, 0.85, 0.93, 0.80, 0.90, 0.88]

        for model, metric in zip(models, metrics):
            results.append(
                ExperimentResult(
                    name=f"exp_{model}",
                    pipeline=Pipeline([]),
                    metrics={"test_accuracy": metric},
                    config={"model": model},
                )
            )

        sorted_results = sorted(
            results, key=lambda r: r.metrics["test_accuracy"], reverse=True
        )
        top_k = evaluator._extract_top_k_by_family(sorted_results, k=3)

        assert len(top_k) == 3
        families = list(top_k.keys())
        assert len(families) == len(set(families))
        assert top_k["RF"].metrics["test_accuracy"] == 0.95
        assert top_k["SVM"].metrics["test_accuracy"] == 0.93
        assert top_k["LR"].metrics["test_accuracy"] == 0.90

    def test_extract_top_k_limited_families(self, evaluator):
        results = []
        for i in range(10):
            results.append(
                ExperimentResult(
                    name=f"exp_{i}",
                    pipeline=Pipeline([]),
                    metrics={"test_accuracy": 0.9 - (i * 0.01)},
                    config={"model": f"Model_{i % 5}"},
                )
            )

        sorted_results = sorted(
            results, key=lambda r: r.metrics["test_accuracy"], reverse=True
        )
        top_k = evaluator._extract_top_k_by_family(sorted_results, k=2)
        assert len(top_k) == 2

    def test_log_best_experiment(self, evaluator):
        result = ExperimentResult(
            name="best_model",
            pipeline=Pipeline([]),
            metrics={"test_accuracy": 0.95},
            config={"model": "RandomForest"},
        )
        evaluator._log_best_experiment(result)

    def test_evaluate_with_single_result(self, evaluator, sample_results):
        single_result = [sample_results[0]]
        evaluator.evaluate(single_result)
        evaluator.context.update_stage_context.assert_called_once()

    def test_evaluate_passes_best_to_hooks(self, evaluator, sample_results):
        evaluator.evaluate(sample_results)
        call_args = evaluator.context.update_stage_context.call_args
        stage_result = call_args[0][1]
        assert stage_result.best_experiment.metrics["test_accuracy"] == 0.95

    def test_sort_results_consistency(self, evaluator):
        results = []
        for i in range(5):
            results.append(
                ExperimentResult(
                    name=f"exp_{i}",
                    pipeline=Pipeline([]),
                    metrics={"test_accuracy": 0.8 + (i * 0.03)},
                )
            )

        sorted_1 = evaluator._sort_results(results)
        sorted_2 = evaluator._sort_results(results)
        assert [r.name for r in sorted_1] == [r.name for r in sorted_2]


class TestEvaluatorWithMultipleScoringMetrics:
    @pytest.fixture
    def multi_metric_context(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_accuracy"
        config.scoring = ["accuracy", "precision"]
        context = Mock(spec=Context)
        context.config = config
        context.update_stage_context = MagicMock()
        return context

    def test_sort_uses_priority_metric(self, multi_metric_context):
        evaluator = ConcreteEvaluator(
            stage=Stages.MODEL_SELECTION, context=multi_metric_context
        )

        results = []
        for acc, prec in [(0.95, 0.80), (0.85, 0.95), (0.90, 0.85)]:
            results.append(
                ExperimentResult(
                    name="exp",
                    pipeline=Pipeline([]),
                    metrics={"test_accuracy": acc, "test_precision": prec},
                )
            )

        sorted_results = evaluator._sort_results(results)
        assert sorted_results[0].metrics["test_accuracy"] == 0.95
        assert sorted_results[0].metrics["test_precision"] == 0.80
