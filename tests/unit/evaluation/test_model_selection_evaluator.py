"""Tests for ModelSelectionEvaluator."""

import pytest
from unittest.mock import Mock, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from app.core.stages.evaluation.evaluators.model_selection_evaluator import (
    ModelSelectionEvaluator,
)
from app.core.experiments import ExperimentResult
from app.core.context.context import Context, StageResult, ProjectConfig
from app.core.enums import Stages


@pytest.fixture
def mock_config():
    config = Mock(spec=ProjectConfig)
    config.priority_metric = "test_accuracy"
    config.scoring = ["accuracy"]
    return config


@pytest.fixture
def mock_context(mock_config):
    context = Mock(spec=Context)
    context.config = mock_config
    context.stage_results = {}
    context.update_stage_context = MagicMock()
    return context


@pytest.fixture
def evaluator(mock_context):
    return ModelSelectionEvaluator(stage=Stages.MODEL_SELECTION, context=mock_context)


@pytest.fixture
def sample_results():
    models = [
        "LogisticRegression",
        "RandomForest",
        "RandomForest",
        "SVM",
        "GradientBoosting",
        "SVM",
    ]
    metrics = [0.85, 0.95, 0.90, 0.92, 0.94, 0.88]
    model_classes = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "GradientBoosting": GradientBoostingClassifier(),
    }
    results = []
    for model_name, metric in zip(models, metrics):
        results.append(
            ExperimentResult(
                name=f"exp_{model_name}_{metric}",
                pipeline=Pipeline([("classifier", model_classes[model_name])]),
                metrics={"test_accuracy": metric},
                config={"model": model_name, "param": f"val_{metric}"},
            )
        )
    return results


class TestModelSelectionEvaluator:
    def test_evaluator_inheritance(self, evaluator):
        from app.core.stages.evaluation.base_evaluator import BaseEvaluator

        assert isinstance(evaluator, BaseEvaluator)

    def test_extract_stage_specific_data(self, evaluator, sample_results):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])

        assert "top_k_models_by_family" in data
        assert "best_model" in data
        assert "best_selector" in data
        assert "total_experiments" in data
        assert "models_by_family" in data
        assert data["best_model"] == "RandomForest"
        assert len(data["top_k_models_by_family"]) <= 3

    def test_extract_top_k_by_family_ensures_diversity(self, evaluator, sample_results):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        top_k = evaluator._extract_top_k_by_family(sorted_results, k=3)
        families = list(top_k.keys())
        assert len(families) == len(set(families))
        assert top_k["RandomForest"].metrics["test_accuracy"] == 0.95
        assert top_k["GradientBoosting"].metrics["test_accuracy"] == 0.94
        assert top_k["SVM"].metrics["test_accuracy"] == 0.92

    def test_get_all_models_by_family(self, evaluator, sample_results):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])
        models_by_family = data["models_by_family"]

        assert "RandomForest" in models_by_family
        assert "SVM" in models_by_family
        assert "LogisticRegression" in models_by_family
        assert len(models_by_family["RandomForest"]) == 2
        assert len(models_by_family["SVM"]) == 2
        assert len(models_by_family["LogisticRegression"]) == 1
        assert len(models_by_family["GradientBoosting"]) == 1

    def test_update_context_creates_stage_result(self, evaluator, sample_results):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        best = sorted_results[0]
        stage_specific_data = evaluator._extract_stage_specific_data(
            sorted_results, best
        )
        evaluator._update_context(sorted_results, best, stage_specific_data)

        assert evaluator.context.update_stage_context.called
        call_args = evaluator.context.update_stage_context.call_args
        stage, stage_result = call_args[0]
        assert stage == Stages.MODEL_SELECTION
        assert stage_result.best_experiment == best
        assert "top_k_models_by_family" in stage_result.metadata

    def test_evaluate_complete_workflow(self, evaluator, sample_results):
        evaluator.evaluate(sample_results)
        assert evaluator.context.update_stage_context.called
        call_args = evaluator.context.update_stage_context.call_args
        stage, stage_result = call_args[0]
        assert isinstance(stage_result, StageResult)
        assert stage_result.name == Stages.MODEL_SELECTION
        assert stage_result.best_experiment is not None

    def test_best_model_identified_correctly(self, evaluator, sample_results):
        evaluator.evaluate(sample_results)
        call_args = evaluator.context.update_stage_context.call_args
        _, stage_result = call_args[0]
        assert stage_result.best_experiment.metrics["test_accuracy"] == 0.95
        assert stage_result.best_experiment.config["model"] == "RandomForest"

    def test_top_k_models_are_from_different_families(self, evaluator, sample_results):
        evaluator.evaluate(sample_results)
        call_args = evaluator.context.update_stage_context.call_args
        _, stage_result = call_args[0]
        top_k = stage_result.metadata["top_k_models_by_family"]
        families = list(top_k.keys())
        assert len(families) == len(set(families))


class TestModelSelectionEvaluatorEdgeCases:
    @pytest.fixture
    def minimal_evaluator(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_accuracy"
        config.scoring = ["accuracy"]
        context = Mock(spec=Context)
        context.config = config
        context.update_stage_context = MagicMock()
        return ModelSelectionEvaluator(
            stage=Stages.MODEL_SELECTION, context=context
        ), context

    def test_single_model_family(self, minimal_evaluator):
        evaluator, context = minimal_evaluator
        results = []
        for i in range(3):
            results.append(
                ExperimentResult(
                    name=f"exp_{i}",
                    pipeline=Pipeline([("classifier", LogisticRegression())]),
                    metrics={"test_accuracy": 0.85 + (i * 0.03)},
                    config={"model": "LogisticRegression"},
                )
            )
        evaluator.evaluate(results)
        call_args = context.update_stage_context.call_args
        _, stage_result = call_args[0]
        assert stage_result.best_experiment is not None
        assert abs(stage_result.best_experiment.metrics["test_accuracy"] - 0.91) < 0.001

    def test_insufficient_families_for_top_3(self, minimal_evaluator):
        evaluator, context = minimal_evaluator
        results = []
        for model, metric in [("LogisticRegression", 0.90), ("RandomForest", 0.95)]:
            results.append(
                ExperimentResult(
                    name=f"exp_{model}",
                    pipeline=Pipeline([("clf", LogisticRegression())]),
                    metrics={"test_accuracy": metric},
                    config={"model": model},
                )
            )
        evaluator.evaluate(results)
        call_args = context.update_stage_context.call_args
        _, stage_result = call_args[0]
        top_k = stage_result.metadata["top_k_models_by_family"]
        assert len(top_k) == 2

    def test_many_model_families(self, minimal_evaluator):
        evaluator, context = minimal_evaluator
        results = []
        for i in range(10):
            results.append(
                ExperimentResult(
                    name=f"exp_{i}",
                    pipeline=Pipeline([("clf", LogisticRegression())]),
                    metrics={"test_accuracy": 0.80 + (i * 0.01)},
                    config={"model": f"Model_Family_{i}"},
                )
            )
        evaluator.evaluate(results)
        call_args = context.update_stage_context.call_args
        _, stage_result = call_args[0]
        top_k = stage_result.metadata["top_k_models_by_family"]
        assert len(top_k) == 3

    def test_models_by_family_count(self, minimal_evaluator):
        evaluator, context = minimal_evaluator
        configs = [
            ("LogisticRegression", 0.85),
            ("LogisticRegression", 0.87),
            ("LogisticRegression", 0.89),
            ("RandomForest", 0.90),
            ("RandomForest", 0.92),
        ]
        results = []
        for model, metric in configs:
            results.append(
                ExperimentResult(
                    name=f"exp_{metric}",
                    pipeline=Pipeline([("clf", LogisticRegression())]),
                    metrics={"test_accuracy": metric},
                    config={"model": model},
                )
            )
        evaluator.evaluate(results)
        call_args = context.update_stage_context.call_args
        _, stage_result = call_args[0]
        models_by_family = stage_result.metadata.get("models_by_family", {})
        assert (
            "LogisticRegression" in models_by_family
            or "RandomForest" in models_by_family
        )


class TestModelSelectionEvaluatorModelFamilyExtraction:
    @pytest.fixture
    def setup_evaluator(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_accuracy"
        config.scoring = ["accuracy"]
        context = Mock(spec=Context)
        context.config = config
        return ModelSelectionEvaluator(stage=Stages.MODEL_SELECTION, context=context)

    def test_model_family_extraction_consistency(self, setup_evaluator):
        evaluator = setup_evaluator
        result = ExperimentResult(
            name="test", pipeline=Pipeline([]), config={"model": "RandomForest"}
        )
        family1 = evaluator._get_model_family(result)
        family2 = evaluator._get_model_family(result)
        assert family1 == family2 == "RandomForest"

    def test_different_families_recognized(self, setup_evaluator):
        evaluator = setup_evaluator
        result1 = ExperimentResult(
            name="test1", pipeline=Pipeline([]), config={"model": "RandomForest"}
        )
        result2 = ExperimentResult(
            name="test2", pipeline=Pipeline([]), config={"model": "SVM"}
        )
        assert evaluator._get_model_family(result1) != evaluator._get_model_family(
            result2
        )
        assert evaluator._get_model_family(result1) == "RandomForest"
        assert evaluator._get_model_family(result2) == "SVM"
