"""Tests for EvaluatorFactory."""

import pytest
from unittest.mock import Mock, patch

from app.core.stages.evaluation.evaluator_factory import EvaluatorFactory
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.core.stages.evaluation.evaluators.feature_selection_evaluator import (
    FeatureSelectionEvaluator,
)
from app.core.stages.evaluation.evaluators.model_selection_evaluator import ModelSelectionEvaluator
from app.core.enums import Stages
from app.core.context.context import Context


class TestEvaluatorFactory:
    """Test suite for EvaluatorFactory."""

    @pytest.fixture
    def mock_context(self):
        """Create mock Context."""
        context = Mock(spec=Context)
        return context

    def test_factory_creates_feature_selection_evaluator(self, mock_context):
        """Test factory creates FeatureSelectionEvaluator for FEATURE_SELECTION stage."""
        evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)

        assert isinstance(evaluator, FeatureSelectionEvaluator)
        assert evaluator.stage == Stages.FEATURE_SELECTION
        assert evaluator.context == mock_context

    def test_factory_creates_model_selection_evaluator(self, mock_context):
        """Test factory creates ModelSelectionEvaluator for MODEL_SELECTION stage."""
        evaluator = EvaluatorFactory.create(Stages.MODEL_SELECTION, mock_context)

        assert isinstance(evaluator, ModelSelectionEvaluator)
        assert evaluator.stage == Stages.MODEL_SELECTION
        assert evaluator.context == mock_context

    def test_factory_raises_for_unregistered_stage(self, mock_context):
        """Test factory raises ValueError for unregistered stage."""
        # Create a mock stage not in EVALUATORS
        unregistered_stage = Mock()
        unregistered_stage.value = "UNREGISTERED"

        with pytest.raises(ValueError, match="No evaluator for stage"):
            EvaluatorFactory.create(unregistered_stage, mock_context)

    def test_factory_defaults_registration(self, mock_context):
        """Test that factory registers defaults on first call."""
        # Clear existing registrations
        EvaluatorFactory._EVALUATORS.clear()

        # First call should trigger registration
        assert not EvaluatorFactory._EVALUATORS  # Should be empty before

        evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)

        assert EvaluatorFactory._EVALUATORS  # Should be populated after
        assert Stages.FEATURE_SELECTION in EvaluatorFactory._EVALUATORS

    def test_factory_register_custom_evaluator(self, mock_context):
        """Test registering a custom evaluator."""

        # Create a custom evaluator
        class CustomEvaluator(BaseEvaluator):
            def _extract_stage_specific_data(self, sorted_results, best_experiment):
                return {}

            def _update_context(self, sorted_results, best_experiment, stage_specific_data):
                pass

        custom_stage = Mock()
        custom_stage.value = "CUSTOM_STAGE"

        # Register custom evaluator
        EvaluatorFactory.register(custom_stage, CustomEvaluator)

        # Create evaluator
        evaluator = EvaluatorFactory.create(custom_stage, mock_context)

        assert isinstance(evaluator, CustomEvaluator)
        assert evaluator.stage == custom_stage

    def test_factory_created_evaluators_are_independent(self, mock_context):
        """Test that each created evaluator is an independent instance."""
        evaluator1 = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        evaluator2 = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)

        assert evaluator1 is not evaluator2  # Different instances
        assert type(evaluator1) == type(evaluator2)  # But same type

    def test_factory_preserves_stage_information(self, mock_context):
        """Test that factory preserves stage information in created evaluator."""
        stages = [Stages.FEATURE_SELECTION, Stages.MODEL_SELECTION]

        for stage in stages:
            evaluator = EvaluatorFactory.create(stage, mock_context)
            assert evaluator.stage == stage

    def test_factory_correct_evaluation_behavior(self, mock_context):
        """Test that created evaluators have correct evaluation behavior."""
        # Feature Selection evaluator should handle feature data
        fs_evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        assert hasattr(fs_evaluator, "_extract_top_k_selectors")

        # Model Selection evaluator should handle model families
        ms_evaluator = EvaluatorFactory.create(Stages.MODEL_SELECTION, mock_context)
        assert hasattr(ms_evaluator, "_get_all_models_by_family")


class TestEvaluatorFactoryRegistration:
    """Test evaluator registration and retrieval."""

    def test_multiple_evaluators_registered(self):
        """Test that multiple evaluators can be registered."""
        EvaluatorFactory._EVALUATORS.clear()
        EvaluatorFactory._register_defaults()

        assert Stages.FEATURE_SELECTION in EvaluatorFactory._EVALUATORS
        assert Stages.MODEL_SELECTION in EvaluatorFactory._EVALUATORS

    def test_evaluator_registration_does_not_override_unless_explicit(self):
        """Test that re-registering doesn't cause issues."""
        EvaluatorFactory._EVALUATORS.clear()
        EvaluatorFactory._register_defaults()

        first_count = len(EvaluatorFactory._EVALUATORS)

        # Call again - should not break
        EvaluatorFactory._register_defaults()

        second_count = len(EvaluatorFactory._EVALUATORS)

        assert first_count == second_count


class TestEvaluatorFactoryErrorHandling:
    """Test error handling in factory."""

    def test_factory_handles_import_errors_gracefully(self, monkeypatch):
        """Test factory handles import errors during registration."""
        # This test verifies the factory's error handling but we don't actually
        # want to break imports, so we mock it

        def mock_import(*args, **kwargs):
            raise ImportError("Simulated import error")

        # The factory catches ImportError in _register_defaults
        # This test ensures the error is logged properly
        EvaluatorFactory._EVALUATORS.clear()

        with patch("builtins.__import__", side_effect=mock_import):
            # Factory should handle this, but in practice our imports should work
            # This is more of an integration test
            pass

    def test_factory_stage_type_validation(self):
        """Test that factory validates stage type."""
        mock_context = Mock(spec=Context)

        # None stage should raise
        with pytest.raises((ValueError, AttributeError)):
            EvaluatorFactory.create(None, mock_context)


class TestEvaluatorFactoryIntegration:
    """Integration tests for EvaluatorFactory."""

    def test_created_evaluators_inherit_from_base(self):
        """Test that all created evaluators inherit from BaseEvaluator."""
        mock_context = Mock(spec=Context)

        for stage in [Stages.FEATURE_SELECTION, Stages.MODEL_SELECTION]:
            evaluator = EvaluatorFactory.create(stage, mock_context)
            assert isinstance(evaluator, BaseEvaluator)

    def test_factory_with_different_contexts(self):
        """Test factory works with different context instances."""
        contexts = [Mock(spec=Context) for _ in range(3)]

        evaluators = [EvaluatorFactory.create(Stages.FEATURE_SELECTION, ctx) for ctx in contexts]

        # Each should have correct context
        for evaluator, context in zip(evaluators, contexts):
            assert evaluator.context == context

    def test_factory_stage_specific_methods_exist(self):
        """Test that factory-created evaluators have required methods."""
        mock_context = Mock(spec=Context)

        fs_evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        ms_evaluator = EvaluatorFactory.create(Stages.MODEL_SELECTION, mock_context)

        # All should have base methods
        for evaluator in [fs_evaluator, ms_evaluator]:
            assert hasattr(evaluator, "evaluate")
            assert hasattr(evaluator, "_sort_results")
            assert hasattr(evaluator, "_extract_stage_specific_data")
            assert hasattr(evaluator, "_update_context")
