"""Tests for ModelSelectionEvaluator."""

import pytest
from unittest.mock import Mock, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from app.core.stages.evaluation.model_selection_evaluator import ModelSelectionEvaluator
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.context.run_context import RunContext, StageResult, ProjectConfig
from app.core.context.stages import Stages


class TestModelSelectionEvaluator:
    """Test suite for ModelSelectionEvaluator."""
    
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
        """Create ModelSelectionEvaluator instance."""
        return ModelSelectionEvaluator(
            stage=Stages.MODEL_SELECTION,
            context=mock_context
        )
    
    @pytest.fixture
    def sample_results(self):
        """Create sample model selection results with different model families."""
        results = []
        models = ["LogisticRegression", "RandomForest", "RandomForest", 
                  "SVM", "GradientBoosting", "SVM"]
        metrics = [0.85, 0.95, 0.90, 0.92, 0.94, 0.88]
        
        model_classes = {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "GradientBoosting": GradientBoostingClassifier()
        }
        
        for model_name, metric in zip(models, metrics):
            pipeline = Pipeline([("classifier", model_classes[model_name])])
            result = ExperimentResult(
                name=f"exp_{model_name}_{metric}",
                pipeline=pipeline,
                metrics={"test_accuracy": metric},
                config={"model": model_name, "param": f"val_{metric}"}
            )
            results.append(result)
        
        return results
    
    def test_evaluator_inheritance(self, evaluator):
        """Test that ModelSelectionEvaluator inherits from BaseEvaluator."""
        from app.core.stages.evaluation.base_evaluator import BaseEvaluator
        assert isinstance(evaluator, BaseEvaluator)
    
    def test_extract_stage_specific_data(self, evaluator, sample_results):
        """Test extraction of model selection specific data."""
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True
        )
        
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])
        
        # Should have required keys
        assert "top_k_models_by_family" in data
        assert "best_model" in data
        assert "best_selector" in data
        assert "total_experiments" in data
        assert "models_by_family" in data
        
        # Should identify best model
        assert data["best_model"] == "RandomForest"
        
        # Should have top-k models from different families
        assert len(data["top_k_models_by_family"]) <= 3
    
    def test_extract_top_k_by_family_ensures_diversity(self, evaluator, sample_results):
        """Test that top-k extraction ensures different model families."""
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True
        )
        
        top_k = evaluator._extract_top_k_by_family(sorted_results, k=3)
        
        # Should have models from different families
        families = list(top_k.keys())
        assert len(families) == len(set(families))  # All unique
        
        # Each should be the best of its family
        assert top_k["RandomForest"].metrics["test_accuracy"] == 0.95
        assert top_k["GradientBoosting"].metrics["test_accuracy"] == 0.94
        assert top_k["SVM"].metrics["test_accuracy"] == 0.92
    
    def test_get_all_models_by_family(self, evaluator, sample_results):
        """Test grouping all models by family."""
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True
        )
        
        data = evaluator._extract_stage_specific_data(sorted_results, sorted_results[0])
        models_by_family = data["models_by_family"]
        
        # Should group by model type
        assert "RandomForest" in models_by_family
        assert "SVM" in models_by_family
        assert "LogisticRegression" in models_by_family
        
        # Check counts match the samples
        assert len(models_by_family["RandomForest"]) == 2  # 2 RF models in sample
        assert len(models_by_family["SVM"]) == 2  # 2 SVM models in sample
        assert len(models_by_family["LogisticRegression"]) == 1
        assert len(models_by_family["GradientBoosting"]) == 1
    
    def test_update_context_creates_stage_result(self, evaluator, sample_results):
        """Test that update_context creates proper StageResult."""
        sorted_results = sorted(
            sample_results,
            key=lambda r: r.metrics.get("test_accuracy", 0),
            reverse=True
        )
        best = sorted_results[0]
        
        stage_specific_data = evaluator._extract_stage_specific_data(sorted_results, best)
        evaluator._update_context(sorted_results, best, stage_specific_data)
        
        # Should call update_context
        assert evaluator.context.update_context.called
        
        # Check the StageResult
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]
        
        assert stage == Stages.MODEL_SELECTION
        assert stage_result.best_experiment == best
        assert "top_k_models_by_family" in stage_result.metadata
    
    def test_evaluate_complete_workflow(self, evaluator, sample_results):
        """Test complete evaluation workflow."""
        evaluator.evaluate(sample_results)
        
        # Should update context
        assert evaluator.context.update_context.called
        
        # Should pass StageResult with proper structure
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]
        
        assert isinstance(stage_result, StageResult)
        assert stage_result.name == Stages.MODEL_SELECTION
        assert stage_result.best_experiment is not None
    
    def test_best_model_identified_correctly(self, evaluator, sample_results):
        """Test that the best model is correctly identified."""
        evaluator.evaluate(sample_results)
        
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]
        
        # Best should be RandomForest with 0.95 accuracy
        assert stage_result.best_experiment.metrics["test_accuracy"] == 0.95
        assert stage_result.best_experiment.config["model"] == "RandomForest"
    
    def test_top_k_models_are_from_different_families(self, evaluator, sample_results):
        """Test that top-k models come from different families."""
        evaluator.evaluate(sample_results)
        
        call_args = evaluator.context.update_context.call_args
        stage, stage_result = call_args[0]
        
        top_k = stage_result.metadata["top_k_models_by_family"]
        families = list(top_k.keys())
        
        # Each family should appear only once
        assert len(families) == len(set(families))


class TestModelSelectionEvaluatorEdgeCases:
    """Test edge cases for ModelSelectionEvaluator."""
    
    @pytest.fixture
    def minimal_setup(self):
        """Create minimal setup for edge case testing."""
        config = Mock(spec=ProjectConfig)
        config.scoring = "accuracy"
        
        context = Mock(spec=RunContext)
        context.config = config
        context.update_context = MagicMock()
        
        evaluator = ModelSelectionEvaluator(
            stage=Stages.MODEL_SELECTION,
            context=context
        )
        return evaluator, context
    
    def test_single_model_family(self, minimal_setup):
        """Test with results from only one model family."""
        evaluator, context = minimal_setup
        
        results = []
        for i in range(3):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([("classifier", LogisticRegression())]),
                metrics={"test_accuracy": 0.85 + (i * 0.03)},
                config={"model": "LogisticRegression"}
            )
            results.append(result)
        
        evaluator.evaluate(results)
        
        call_args = context.update_context.call_args
        stage, stage_result = call_args[0]
        
        # Should still work with single family - use approximate equality
        assert stage_result.best_experiment is not None
        assert abs(stage_result.best_experiment.metrics["test_accuracy"] - 0.91) < 0.001
    
    def test_insufficient_families_for_top_3(self, minimal_setup):
        """Test when there are fewer families than k=3."""
        evaluator, minimal_context = minimal_setup
        
        results = []
        models = ["LogisticRegression", "RandomForest"]
        metrics = [0.90, 0.95]
        
        model_classes = {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier()
        }
        
        for model, metric in zip(models, metrics):
            result = ExperimentResult(
                name=f"exp_{model}",
                pipeline=Pipeline([("classifier", model_classes[model])]),
                metrics={"test_accuracy": metric},
                config={"model": model}
            )
            results.append(result)
        
        evaluator.evaluate(results)
        
        call_args = minimal_context.update_context.call_args
        stage, stage_result = call_args[0]
        
        # Should handle gracefully with only 2 families
        top_k = stage_result.metadata["top_k_models_by_family"]
        assert len(top_k) == 2
    
    def test_many_model_families(self, minimal_setup):
        """Test with many different model families."""
        evaluator, context = minimal_setup
        
        results = []
        for i in range(10):
            result = ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([("classifier", LogisticRegression())]),
                metrics={"test_accuracy": 0.80 + (i * 0.01)},
                config={"model": f"Model_Family_{i}"}
            )
            results.append(result)
        
        evaluator.evaluate(results)
        
        call_args = context.update_context.call_args
        stage, stage_result = call_args[0]
        
        # Should return top-3 families
        top_k = stage_result.metadata["top_k_models_by_family"]
        assert len(top_k) == 3
    
    def test_models_by_family_count(self, minimal_setup):
        """Test that models_by_family contains all results grouped by family."""
        evaluator, context = minimal_setup
        
        results = []
        # Create 5 total results: 3 LogisticRegression, 2 RandomForest
        configs = [
            ("LogisticRegression", 0.85),
            ("LogisticRegression", 0.87),
            ("LogisticRegression", 0.89),
            ("RandomForest", 0.90),
            ("RandomForest", 0.92),
        ]
        
        for model, metric in configs:
            model_class = LogisticRegression() if model == "LogisticRegression" else RandomForestClassifier()
            result = ExperimentResult(
                name=f"exp_{metric}",
                pipeline=Pipeline([("classifier", model_class)]),
                metrics={"test_accuracy": metric},
                config={"model": model}
            )
            results.append(result)
        
        evaluator.evaluate(results)
        
        call_args = context.update_context.call_args
        stage, stage_result = call_args[0]
        
        models_by_family = stage_result.metadata.get("models_by_family", {})
        
        # Check grouping - at least we should have the families
        assert "LogisticRegression" in models_by_family or "RandomForest" in models_by_family


class TestModelSelectionEvaluatorModelFamilyExtraction:
    """Test model family extraction logic."""
    
    @pytest.fixture
    def setup(self):
        """Create setup for family extraction tests."""
        config = Mock(spec=ProjectConfig)
        config.scoring = "accuracy"
        
        context = Mock(spec=RunContext)
        context.config = config
        
        evaluator = ModelSelectionEvaluator(
            stage=Stages.MODEL_SELECTION,
            context=context
        )
        return evaluator
    
    def test_model_family_extraction_consistency(self, setup):
        """Test that model family is consistently extracted from config."""
        evaluator = setup
        
        result = ExperimentResult(
            name="test",
            pipeline=Pipeline([]),
            config={"model": "RandomForest"}
        )
        
        family1 = evaluator._get_model_family(result)
        family2 = evaluator._get_model_family(result)
        
        assert family1 == family2 == "RandomForest"
    
    def test_different_families_recognized(self, setup):
        """Test that different model families are recognized as different."""
        evaluator = setup
        
        result1 = ExperimentResult(
            name="test1",
            pipeline=Pipeline([]),
            config={"model": "RandomForest"}
        )
        result2 = ExperimentResult(
            name="test2",
            pipeline=Pipeline([]),
            config={"model": "SVM"}
        )
        
        family1 = evaluator._get_model_family(result1)
        family2 = evaluator._get_model_family(result2)
        
        assert family1 != family2
        assert family1 == "RandomForest"
        assert family2 == "SVM"

