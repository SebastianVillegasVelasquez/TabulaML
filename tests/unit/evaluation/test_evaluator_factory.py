"""Tests for EvaluatorFactory."""

import pytest
from unittest.mock import Mock, MagicMock

from app.core.stages.evaluation.evaluator_factory import EvaluatorFactory
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.core.stages.evaluation.evaluators.feature_selection_evaluator import (
    FeatureSelectionEvaluator,
)
from app.core.stages.evaluation.evaluators.model_selection_evaluator import (
    ModelSelectionEvaluator,
)
from app.core.enums import Stages
from app.core.context.context import Context, ProjectConfig


@pytest.fixture
def mock_context():
    config = Mock(spec=ProjectConfig)
    config.priority_metric = "test_f1"
    config.scoring = ["f1"]
    context = Mock(spec=Context)
    context.config = config
    context.update_stage_context = MagicMock()
    return context


class TestEvaluatorFactory:
    def test_factory_creates_feature_selection_evaluator(self, mock_context):
        evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        assert isinstance(evaluator, FeatureSelectionEvaluator)
        assert evaluator.stage == Stages.FEATURE_SELECTION
        assert evaluator.context == mock_context

    def test_factory_creates_model_selection_evaluator(self, mock_context):
        evaluator = EvaluatorFactory.create(Stages.MODEL_SELECTION, mock_context)
        assert isinstance(evaluator, ModelSelectionEvaluator)
        assert evaluator.stage == Stages.MODEL_SELECTION
        assert evaluator.context == mock_context

    def test_factory_raises_for_unregistered_stage(self, mock_context):
        unregistered_stage = Mock()
        unregistered_stage.value = "UNREGISTERED"
        with pytest.raises(ValueError, match="No evaluator for stage"):
            EvaluatorFactory.create(unregistered_stage, mock_context)

    def test_factory_defaults_registration(self, mock_context):
        EvaluatorFactory._EVALUATORS.clear()
        assert not EvaluatorFactory._EVALUATORS
        EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        assert EvaluatorFactory._EVALUATORS
        assert Stages.FEATURE_SELECTION in EvaluatorFactory._EVALUATORS

    def test_factory_register_custom_evaluator(self, mock_context):
        class CustomEvaluator(BaseEvaluator):
            def _extract_stage_specific_data(self, sorted_results, best_experiment):
                return {}

            def _update_context(
                self, sorted_results, best_experiment, stage_specific_data
            ):
                pass

        custom_stage = Mock()
        custom_stage.value = "CUSTOM_STAGE"
        EvaluatorFactory.register(custom_stage, CustomEvaluator)
        evaluator = EvaluatorFactory.create(custom_stage, mock_context)
        assert isinstance(evaluator, CustomEvaluator)
        assert evaluator.stage == custom_stage

    def test_factory_created_evaluators_are_independent(self, mock_context):
        evaluator1 = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        evaluator2 = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        assert evaluator1 is not evaluator2
        assert type(evaluator1) is type(evaluator2)

    def test_factory_preserves_stage_information(self, mock_context):
        for stage in [Stages.FEATURE_SELECTION, Stages.MODEL_SELECTION]:
            evaluator = EvaluatorFactory.create(stage, mock_context)
            assert evaluator.stage == stage

    def test_factory_correct_evaluation_behavior(self, mock_context):
        fs_evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        assert hasattr(fs_evaluator, "_extract_top_k_chain_selectors")

        ms_evaluator = EvaluatorFactory.create(Stages.MODEL_SELECTION, mock_context)
        assert hasattr(ms_evaluator, "_get_all_models_by_family")


class TestEvaluatorFactoryRegistration:
    def test_multiple_evaluators_registered(self):
        EvaluatorFactory._EVALUATORS.clear()
        EvaluatorFactory._register_defaults()
        assert Stages.FEATURE_SELECTION in EvaluatorFactory._EVALUATORS
        assert Stages.MODEL_SELECTION in EvaluatorFactory._EVALUATORS

    def test_evaluator_registration_does_not_override_unless_explicit(self):
        EvaluatorFactory._EVALUATORS.clear()
        EvaluatorFactory._register_defaults()
        first_count = len(EvaluatorFactory._EVALUATORS)
        EvaluatorFactory._register_defaults()
        second_count = len(EvaluatorFactory._EVALUATORS)
        assert first_count == second_count


class TestEvaluatorFactoryErrorHandling:
    def test_factory_stage_type_validation(self):
        config = Mock(spec=ProjectConfig)
        config.priority_metric = "test_f1"
        mock_context = Mock(spec=Context)
        mock_context.config = config
        with pytest.raises((ValueError, AttributeError)):
            EvaluatorFactory.create(None, mock_context)


class TestEvaluatorFactoryIntegration:
    def test_created_evaluators_inherit_from_base(self, mock_context):
        for stage in [Stages.FEATURE_SELECTION, Stages.MODEL_SELECTION]:
            evaluator = EvaluatorFactory.create(stage, mock_context)
            assert isinstance(evaluator, BaseEvaluator)

    def test_factory_with_different_contexts(self):
        contexts = []
        for _ in range(3):
            config = Mock(spec=ProjectConfig)
            config.priority_metric = "test_f1"
            ctx = Mock(spec=Context)
            ctx.config = config
            contexts.append(ctx)

        evaluators = [
            EvaluatorFactory.create(Stages.FEATURE_SELECTION, ctx) for ctx in contexts
        ]
        for evaluator, context in zip(evaluators, contexts):
            assert evaluator.context == context

    def test_factory_stage_specific_methods_exist(self, mock_context):
        fs_evaluator = EvaluatorFactory.create(Stages.FEATURE_SELECTION, mock_context)
        ms_evaluator = EvaluatorFactory.create(Stages.MODEL_SELECTION, mock_context)

        for evaluator in [fs_evaluator, ms_evaluator]:
            assert hasattr(evaluator, "evaluate")
            assert hasattr(evaluator, "_sort_results")
            assert hasattr(evaluator, "_extract_stage_specific_data")
            assert hasattr(evaluator, "_update_context")
