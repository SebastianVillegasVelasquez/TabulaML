"""Tests for FeatureSelectionEvaluator."""

from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.context.context import Context, StageResult, ProjectConfig
from app.core.enums import Stages
from app.core.experiments import ExperimentResult
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.core.stages.evaluation.evaluators.feature_selection_evaluator import (
    FeatureSelectionEvaluator,
)


class TestFeatureSelectionEvaluator:
    def test_evaluator_inheritance(self, evaluator):
        assert isinstance(evaluator, BaseEvaluator)

    def test_extract_stage_specific_data_returns_list(self, evaluator, sample_results):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])
        assert isinstance(data, list)

    def test_extract_stage_specific_data_top_k(self, evaluator, sample_results):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])
        assert len(data) <= 3

    def test_extract_stage_specific_data_has_required_keys(
        self, evaluator, sample_results
    ):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])
        for item in data:
            assert "selectors" in item
            assert "model" in item
            assert "model_type" in item
            assert "model_based" in item
            assert "selected_features" in item

    def test_extract_top_k_chain_selectors_returns_list(
        self, evaluator, sample_results
    ):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        top_k = evaluator._extract_top_k_chain_selectors(sorted_results, k=3)
        assert isinstance(top_k, list)
        assert len(top_k) <= 3

    def test_extract_top_k_chain_selectors_selected_features(
        self, evaluator, sample_results
    ):
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True,
        )
        top_k = evaluator._extract_top_k_chain_selectors(sorted_results, k=3)
        for item in top_k:
            assert "selected_features" in item

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
        assert stage == Stages.FEATURE_SELECTION
        assert stage_result.best_experiment == best
        assert isinstance(stage_result.metadata, list)

    def test_evaluate_complete_workflow(self, evaluator, sample_results):
        evaluator.evaluate(sample_results)
        assert evaluator.context.update_stage_context.called

        call_args = evaluator.context.update_stage_context.call_args
        stage, stage_result = call_args[0]
        assert isinstance(stage_result, StageResult)
        assert stage_result.name == Stages.FEATURE_SELECTION
        assert stage_result.best_experiment is not None

    def test_best_experiment_is_highest_metric(self, evaluator, sample_results):
        evaluator.evaluate(sample_results)
        call_args = evaluator.context.update_stage_context.call_args
        _, stage_result = call_args[0]
        assert stage_result.best_experiment.metrics["test_accuracy"] == 0.95


class TestFeatureSelectionEvaluatorEdgeCases:
    @pytest.fixture
    def minimal_evaluator(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_accuracy"
        config.scoring = ["accuracy"]
        context = Mock(spec=Context)
        context.config = config
        context.update_stage_context = MagicMock()
        return FeatureSelectionEvaluator(
            stage=Stages.FEATURE_SELECTION, context=context
        ), context

    def test_empty_selected_features_in_metadata(self, minimal_evaluator):
        evaluator, context = minimal_evaluator
        result = ExperimentResult(
            name="test",
            pipeline=Pipeline([]),
            metrics={"test_accuracy": 0.9},
            config={"selector": "test", "predictor": "test"},
            metadata={
                "selectors": ["test"],
                "model": "LR",
                "model_type": "LINEAR",
                "model_based": "LINEAR",
            },
            selected_features=[],
        )
        evaluator.evaluate([result])
        assert context.update_stage_context.called

    def test_multiple_results_same_selector(self, minimal_evaluator):
        evaluator, context = minimal_evaluator
        results = []
        for i in range(3):
            results.append(
                ExperimentResult(
                    name=f"exp_{i}",
                    pipeline=Pipeline([("feature_selection", SelectKBest(f_classif))]),
                    metrics={"test_accuracy": 0.85 + (i * 0.03)},
                    config={"selector": "SelectKBest", "predictor": "LR"},
                    metadata={
                        "selectors": ["SelectKBest"],
                        "model": "LR",
                        "model_type": "LINEAR",
                        "model_based": "LINEAR",
                    },
                    selected_features=["f1"],
                )
            )
        evaluator.evaluate(results)
        assert context.update_stage_context.called
        call_args = context.update_stage_context.call_args
        _, stage_result = call_args[0]
        assert abs(stage_result.best_experiment.metrics["test_accuracy"] - 0.91) < 0.001


class TestFeatureSelectionEvaluatorWithRealPipeline:
    def test_with_selectkbest_pipeline(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_accuracy"
        config.scoring = ["accuracy"]
        context = Mock(spec=Context)
        context.config = config
        context.update_stage_context = MagicMock()

        evaluator = FeatureSelectionEvaluator(
            stage=Stages.FEATURE_SELECTION, context=context
        )

        pipeline = Pipeline(
            [
                ("feature_selection", SelectKBest(f_classif, k=2)),
                ("classifier", LogisticRegression()),
            ]
        )
        X = pd.DataFrame(
            {
                "f1": np.random.randn(20),
                "f2": np.random.randn(20),
                "f3": np.random.randn(20),
                "f4": np.random.randn(20),
            }
        )
        y = np.random.randint(0, 2, 20)
        pipeline.fit(X, y)

        result = ExperimentResult(
            name="real_test",
            pipeline=pipeline,
            metrics={"test_accuracy": 0.95},
            config={"selector": "SelectKBest", "predictor": "LogisticRegression"},
            metadata={
                "selectors": ["SelectKBest"],
                "model": "LogisticRegression",
                "model_type": "LINEAR",
                "model_based": "LINEAR",
            },
            selected_features=["f1", "f2"],
        )

        evaluator.evaluate([result])
        assert context.update_stage_context.called
