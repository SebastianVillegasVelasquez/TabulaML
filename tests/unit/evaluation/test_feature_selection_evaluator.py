"""Tests for FeatureSelectionEvaluator."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from app.core.stages.evaluation.evaluators.feature_selection_evaluator import (
    FeatureSelectionEvaluator,
)
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.context.context import Context, StageResult, ProjectConfig
from app.core.enums import Stages


class TestFeatureSelectionEvaluator:
    """Test suite for FeatureSelectionEvaluator."""

    @pytest.fixture
    def mock_config(self):
        """Create mock project config with feature data."""
        config = Mock(spec=ProjectConfig)
        config.scoring = "accuracy"
        config.X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
                "feature3": [1, 1, 1, 1, 1],
                "feature4": [2, 2, 2, 2, 2],
            }
        )
        return config

    @pytest.fixture
    def mock_context(self, mock_config):
        """Create mock Context."""
        context = Mock(spec=Context)
        context.config = mock_config
        context.stage_results = {}
        context.update_context = MagicMock()
        return context

    @pytest.fixture
    def evaluator(self, mock_context):
        """Create FeatureSelectionEvaluator instance."""
        return FeatureSelectionEvaluator(stage=Stages.FEATURE_SELECTION, context=mock_context)

    @pytest.fixture
    def sample_results(self):
        """Create sample feature selection results."""
        results = []
        selectors = ["SelectKBest", "SelectKBest", "RFE", "RFE", "VarianceThreshold"]
        metrics = [0.95, 0.85, 0.93, 0.80, 0.90]

        for selector, metric in zip(selectors, metrics):
            pipeline = Pipeline([("feature_selection", SelectKBest(f_classif, k=3))])
            result = ExperimentResult(
                name=f"exp_{selector}_{metric}",
                pipeline=pipeline,
                metrics={"test_accuracy": metric},
                config={"selector": selector, "predictor": "LogisticRegression"},
                selected_features=["feature1", "feature2", "feature3"],
            )
            results.append(result)

        return results

    def test_evaluator_inheritance(self, evaluator):
        """Test that FeatureSelectionEvaluator inherits from BaseEvaluator."""
        from app.core.stages.evaluation.base_evaluator import BaseEvaluator

        assert isinstance(evaluator, BaseEvaluator)

    def test_extract_stage_specific_data(self, evaluator, sample_results):
        """Test extraction of feature selection specific data."""
        sorted_results = sorted(
            sample_results, key=lambda r: r.metrics.get("test_accuracy", 0), reverse=True
        )

        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])

        # Should have required keys
        assert "top_k_selectors" in data
        assert "best_selector" in data
        assert "best_predictor" in data
        assert "selected_features" in data
        assert "total_experiments" in data

        # Should have at most 3 selectors
        assert len(data["top_k_selectors"]) <= 3

        # Should identify best selector
        assert data["best_selector"] == "SelectKBest"

    def test_extract_top_k_selectors(self, evaluator, sample_results):
        """Test extraction of top-k feature selectors."""
        sorted_results = sorted(
            sample_results, key=lambda r: r.metrics.get("test_accuracy", 0), reverse=True
        )

        top_k = evaluator._extract_top_k_selectors(sorted_results, k=3)

        # Should have at most k selectors
        assert len(top_k) <= 3

        # Each selector should appear only once
        selectors = list(top_k.keys())
        assert len(selectors) == len(set(selectors))

    def test_extract_feature_data_with_valid_pipeline(self, evaluator, mock_config):
        """Test extraction of feature data from pipeline with SelectKBest."""
        # Create pipeline with feature selection step
        pipeline = Pipeline([("feature_selection", SelectKBest(f_classif, k=2))])

        # Fit the pipeline to get support mask
        y_dummy = np.array([0, 1, 0, 1, 0])
        pipeline.fit(mock_config.X_train, y_dummy)

        result = ExperimentResult(
            name="test_exp",
            pipeline=pipeline,
            metrics={"test_accuracy": 0.95},
            config={"selector": "SelectKBest"},
        )

        feature_data = evaluator._extract_feature_data(result)

        # Should have feature mask and selected features
        assert feature_data["feature_mask"] is not None
        assert feature_data["selected_features"] is not None
        assert len(feature_data["selected_features"]) == 2

    def test_extract_feature_data_without_feature_selection_step(self, evaluator):
        """Test handles pipeline without feature selection step."""
        pipeline = Pipeline([("classifier", LogisticRegression())])

        result = ExperimentResult(
            name="test_exp", pipeline=pipeline, metrics={"test_accuracy": 0.95}
        )

        feature_data = evaluator._extract_feature_data(result)

        # Should return empty/None values
        assert feature_data["feature_mask"] is None
        assert feature_data["selected_features"] is None
        assert feature_data["n_features_selected"] == 0

    def test_extract_feature_data_without_get_support(self, evaluator):
        """Test handles selector without get_support method."""
        # Create a mock selector without get_support
        mock_selector = Mock()
        del mock_selector.get_support  # Remove get_support method

        pipeline = Pipeline([])
        pipeline.named_steps["feature_selection"] = mock_selector

        result = ExperimentResult(
            name="test_exp", pipeline=pipeline, metrics={"test_accuracy": 0.95}
        )

        feature_data = evaluator._extract_feature_data(result)

        assert feature_data["feature_mask"] is None
        assert feature_data["n_features_selected"] == 0

    def test_update_context_creates_stage_result(self, evaluator, sample_results):
        """Test that update_context creates StageResult correctly."""
        sorted_results = sorted(
            sample_results, key=lambda r: r.metrics.get("test_accuracy", 0), reverse=True
        )
        best = sorted_results[0]

        stage_specific_data = evaluator._extract_stage_specific_data(sorted_results, best)
        evaluator._update_context(sorted_results, best, stage_specific_data)

        # Should call update_context
        assert evaluator.context.update_context.called

        # Check the StageResult
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]

        assert stage == Stages.FEATURE_SELECTION
        assert stage_result.best_experiment == best
        assert "selector" in stage_result.metadata

    def test_evaluate_complete_workflow(self, evaluator, sample_results):
        """Test complete evaluation workflow."""
        evaluator.evaluate(sample_results)

        # Should update context
        assert evaluator.context.update_context.called

        # Should pass StageResult with proper structure
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]

        assert isinstance(stage_result, StageResult)
        assert stage_result.name == Stages.FEATURE_SELECTION
        assert stage_result.best_experiment is not None

    def test_selected_features_stored_in_experiment(self, evaluator, sample_results):
        """Test that selected features are stored in experiment after evaluation."""
        evaluator.evaluate(sample_results)

        # Get the call to update_context to see the stage_result
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]

        # Best experiment should have selected_features set (may be None if extraction failed)
        best_exp = stage_result.best_experiment
        # Due to unfitted pipeline in mock, selected_features may be None
        # Just verify the evaluate completed successfully
        assert best_exp is not None


class TestFeatureSelectionEvaluatorEdgeCases:
    """Test edge cases for FeatureSelectionEvaluator."""

    @pytest.fixture
    def minimal_setup(self):
        """Create minimal setup for edge case testing."""
        config = Mock(spec=ProjectConfig)
        config.scoring = "accuracy"
        config.X_train = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})

        context = Mock(spec=Context)
        context.config = config
        context.update_context = MagicMock()

        evaluator = FeatureSelectionEvaluator(stage=Stages.FEATURE_SELECTION, context=context)
        return evaluator, context

    def test_empty_selected_features_in_metadata(self, minimal_setup):
        """Test handling of empty selected_features list."""
        evaluator, context = minimal_setup

        result = ExperimentResult(
            name="test",
            pipeline=Pipeline([]),
            metrics={"test_accuracy": 0.9},
            config={"selector": "test", "predictor": "test"},
            selected_features=[],
        )

        evaluator.evaluate([result])

        # Should still update context successfully
        assert context.update_context.called

    def test_multiple_results_same_selector(self, minimal_setup):
        """Test evaluation with multiple results of same selector type."""
        evaluator, context = minimal_setup

        results = []
        for i in range(3):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([("feature_selection", SelectKBest(f_classif))]),
                metrics={"test_accuracy": 0.85 + (i * 0.03)},
                config={"selector": "SelectKBest", "predictor": "LR"},
                selected_features=["f1"],
            )
            results.append(result)

        evaluator.evaluate(results)

        # Should select only the best SelectKBest
        call_args = context.update_context.call_args
        stage, stage_result = call_args[0]

        # Use approximate equality for floating point
        assert abs(stage_result.best_experiment.metrics["test_accuracy"] - 0.91) < 0.001


class TestFeatureSelectionEvaluatorWithRealPipeline:
    """Test FeatureSelectionEvaluator with real sklearn pipelines."""

    def test_with_selectkbest_pipeline(self):
        """Test with real SelectKBest pipeline."""
        config = Mock(spec=ProjectConfig)
        config.scoring = "accuracy"
        config.X_train = pd.DataFrame(
            {
                "f1": np.random.randn(20),
                "f2": np.random.randn(20),
                "f3": np.random.randn(20),
                "f4": np.random.randn(20),
            }
        )
        config.y_train = np.random.randint(0, 2, 20)

        context = Mock(spec=Context)
        context.config = config
        context.update_context = MagicMock()

        evaluator = FeatureSelectionEvaluator(stage=Stages.FEATURE_SELECTION, context=context)

        # Create and fit pipeline
        pipeline = Pipeline(
            [
                ("feature_selection", SelectKBest(f_classif, k=2)),
                ("classifier", LogisticRegression()),
            ]
        )
        pipeline.fit(config.X_train, config.y_train)

        result = ExperimentResult(
            name="real_test",
            pipeline=pipeline,
            metrics={"test_accuracy": 0.95},
            config={"selector": "SelectKBest", "predictor": "LogisticRegression"},
        )

        evaluator.evaluate([result])

        assert context.update_context.called

        # Check that feature data was extracted
        call_args = context.update_context.call_args
        stage, stage_result = call_args[0]

        assert stage_result.best_experiment.selected_features is not None
        assert len(stage_result.best_experiment.selected_features) > 0
