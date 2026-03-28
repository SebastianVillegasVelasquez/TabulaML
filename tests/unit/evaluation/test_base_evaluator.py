"""Tests for BaseEvaluator abstract class and template method pattern."""

import pytest
from unittest.mock import Mock, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from app.core.enums.stages import Stages
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.context.run_context import RunContext, StageResult, ProjectConfig


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation for testing BaseEvaluator."""
    
    def _extract_stage_specific_data(self, sorted_results, best_experiment):
        return {"test_data": "test_value", "best_name": best_experiment.name}
    
    def _update_context(self, sorted_results, best_experiment, stage_specific_data):
        stage_result = StageResult(
            name=self.stage,
            best_experiment=best_experiment,
            metadata=stage_specific_data
        )
        self.context.update_context(self.stage, stage_result)


class TestBaseEvaluator:
    """Test suite for BaseEvaluator template method pattern."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock project config."""
        config = Mock(spec=ProjectConfig)
        config.scoring = "accuracy"
        return config
    
    @pytest.fixture
    def mock_context(self, mock_config):
        """Create mock RunContext."""
        context = Mock(spec=RunContext)
        context.config = mock_config
        context.stage_results = {}
        context.update_context = MagicMock()
        return context
    
    @pytest.fixture
    def evaluator(self, mock_context):
        """Create concrete evaluator instance."""
        return ConcreteEvaluator(stage=Stages.MODEL_SELECTION, context=mock_context)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample experiment results."""
        results = []
        for i, acc in enumerate([0.85, 0.95, 0.90]):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([("classifier", LogisticRegression())]),
                metrics={f"test_accuracy": acc},
                config={"model": f"Model_{i % 2}", "param": f"value_{i}"}
            )
            results.append(result)
        return results
    
    def test_evaluator_initialization(self, mock_context):
        """Test evaluator initializes with correct stage and context."""
        evaluator = ConcreteEvaluator(
            stage=Stages.FEATURE_SELECTION,
            context=mock_context
        )
        assert evaluator.stage == Stages.FEATURE_SELECTION
        assert evaluator.context == mock_context
        assert evaluator.config == mock_context.config
    
    def test_evaluate_with_empty_results(self, evaluator):
        """Test evaluate handles empty results gracefully."""
        # Should not raise exception
        evaluator.evaluate([])
        # Test passes if no exception is raised
    
    def test_evaluate_calls_template_methods(self, evaluator, sample_results):
        """Test that evaluate calls all template method steps."""
        # Spy on the methods
        evaluator._sort_results = MagicMock(return_value=sample_results)
        evaluator._extract_stage_specific_data = MagicMock(
            return_value={"test": "data"}
        )
        evaluator._update_context = MagicMock()
        
        evaluator.evaluate(sample_results)
        
        # Verify all methods were called
        evaluator._sort_results.assert_called_once()
        evaluator._extract_stage_specific_data.assert_called_once()
        evaluator._update_context.assert_called_once()
    
    def test_evaluate_orders_steps_correctly(self, evaluator, sample_results):
        """Test that evaluate executes steps in correct order."""
        call_order = []
        
        def track_sort(*args, **kwargs):
            call_order.append("sort")
            return sample_results
        
        def track_extract(*args, **kwargs):
            call_order.append("extract")
            return {"data": "value"}
        
        def track_update(*args, **kwargs):
            call_order.append("update")
        
        evaluator._sort_results = Mock(side_effect=track_sort)
        evaluator._extract_stage_specific_data = Mock(side_effect=track_extract)
        evaluator._update_context = Mock(side_effect=track_update)
        
        evaluator.evaluate(sample_results)
        
        assert call_order == ["sort", "extract", "update"]
    
    def test_sort_results_by_accuracy_max_mode(self, evaluator, sample_results):
        """Test sorting results by accuracy (max mode)."""
        sorted_results = evaluator._sort_results(sample_results)
        
        # Should be sorted in descending order (best first)
        assert sorted_results[0].metrics["test_accuracy"] == 0.95
        assert sorted_results[1].metrics["test_accuracy"] == 0.90
        assert sorted_results[2].metrics["test_accuracy"] == 0.85
    
    def test_sort_results_with_list_scoring(self, evaluator, sample_results):
        """Test sorting with multiple scoring metrics."""
        evaluator.config.scoring = ["accuracy", "precision"]
        
        sorted_results = evaluator._sort_results(sample_results)
        
        # Should use first metric from list
        assert sorted_results[0].metrics["test_accuracy"] == 0.95
    
    def test_get_model_family_from_config(self, evaluator):
        """Test extraction of model family from experiment config."""
        result = ExperimentResult(
            name="test",
            pipeline=Pipeline([]),
            config={"model": "RandomForest"}
        )
        
        family = evaluator._get_model_family(result)
        
        assert family == "RandomForest"
    
    def test_get_model_family_unknown_model(self, evaluator):
        """Test handles missing model in config."""
        result = ExperimentResult(
            name="test",
            pipeline=Pipeline([]),
            config={}
        )
        
        family = evaluator._get_model_family(result)
        
        assert family == "unknown"
    
    def test_extract_top_k_by_family_single_per_family(self, evaluator):
        """Test extracting top-k results groups by family (one per family)."""
        results = []
        models = ["RF", "RF", "SVM", "SVM", "LR", "LR"]
        metrics = [0.95, 0.85, 0.93, 0.80, 0.90, 0.88]
        
        for model, metric in zip(models, metrics):
            result = ExperimentResult(
                name=f"exp_{model}",
                pipeline=Pipeline([]),
                metrics={"test_accuracy": metric},
                config={"model": model}
            )
            results.append(result)
        
        # Pre-sort by accuracy
        sorted_results = sorted(
            results,
            key=lambda r: r.metrics["test_accuracy"],
            reverse=True
        )
        
        top_k = evaluator._extract_top_k_by_family(sorted_results, k=3)
        
        # Should have at most 3 families
        assert len(top_k) == 3
        
        # Each family should appear only once (the best one)
        families = list(top_k.keys())
        assert len(families) == len(set(families))
        
        # Check that best of each family is kept
        assert top_k["RF"].metrics["test_accuracy"] == 0.95  # Best RF
        assert top_k["SVM"].metrics["test_accuracy"] == 0.93  # Best SVM
        assert top_k["LR"].metrics["test_accuracy"] == 0.90  # Best LR
    
    def test_extract_top_k_limited_families(self, evaluator):
        """Test k parameter limits the number of families returned."""
        results = []
        for i in range(10):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([]),
                metrics={"test_accuracy": 0.9 - (i * 0.01)},
                config={"model": f"Model_{i % 5}"}
            )
            results.append(result)
        
        sorted_results = sorted(
            results,
            key=lambda r: r.metrics["test_accuracy"],
            reverse=True
        )
        
        top_k = evaluator._extract_top_k_by_family(sorted_results, k=2)
        
        assert len(top_k) == 2
    
    def test_log_best_experiment(self, evaluator):
        """Test logging of best experiment information."""
        result = ExperimentResult(
            name="best_model",
            pipeline=Pipeline([]),
            metrics={"test_accuracy": 0.95},
            config={"model": "RandomForest"}
        )
        
        # Should not raise - just verify it can be called
        evaluator._log_best_experiment(result)
    
    def test_evaluate_with_single_result(self, evaluator, sample_results):
        """Test evaluate with a single result."""
        single_result = [sample_results[0]]
        
        evaluator.evaluate(single_result)
        
        # Should update context with single best result
        evaluator.context.update_context.assert_called_once()
    
    def test_evaluate_passes_best_to_hooks(self, evaluator, sample_results):
        """Test that best experiment is correctly identified and passed to hooks."""
        evaluator.evaluate(sample_results)
        
        # The best should be the one with highest accuracy (0.95)
        call_args = evaluator.context.update_context.call_args
        stage_result = call_args[0][1]
        
        assert stage_result.best_experiment.metrics["test_accuracy"] == 0.95
    
    def test_sort_results_consistency(self, evaluator):
        """Test that sorting is consistent across multiple calls."""
        results = []
        for i in range(5):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([]),
                metrics={"test_accuracy": 0.8 + (i * 0.03)}
            )
            results.append(result)
        
        sorted_1 = evaluator._sort_results(results)
        sorted_2 = evaluator._sort_results(results)
        
        assert [r.name for r in sorted_1] == [r.name for r in sorted_2]


class TestEvaluatorWithMultipleScoringMetrics:
    """Test evaluator with multiple scoring metrics."""
    
    @pytest.fixture
    def multi_metric_context(self):
        """Create context with multiple scoring metrics."""
        config = Mock(spec=ProjectConfig)
        config.scoring = ["accuracy", "precision"]
        
        context = Mock(spec=RunContext)
        context.config = config
        context.update_context = MagicMock()
        return context
    
    def test_sort_uses_first_metric_from_list(self, multi_metric_context):
        """Test that first metric is used when multiple are provided."""
        evaluator = ConcreteEvaluator(
            stage=Stages.MODEL_SELECTION,
            context=multi_metric_context
        )
        
        results = []
        for acc, prec in [(0.95, 0.80), (0.85, 0.95), (0.90, 0.85)]:
            result = ExperimentResult(
                name=f"exp",
                pipeline=Pipeline([]),
                metrics={"test_accuracy": acc, "test_precision": prec}
            )
            results.append(result)
        
        sorted_results = evaluator._sort_results(results)
        
        # Should use accuracy (first metric), not precision
        assert sorted_results[0].metrics["test_accuracy"] == 0.95
        assert sorted_results[0].metrics["test_precision"] == 0.80


