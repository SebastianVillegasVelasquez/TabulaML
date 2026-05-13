from mypy.nodes import Enum, Callable

from app.core.enums import Stages
from app.core.stages.evaluation.evaluators.fine_tuning_evaluation import (
    FineTuningEvaluator,
)
from app.core.stages.evaluation.evaluators.model_ensemble_evaluator import (
    ModelEnsembleEvaluator,
)
from app.utils.logger import logger
from app.core.stages.evaluation import BaseEvaluator


class EvaluatorFactory:
    """Factory for creating stage-specific evaluators."""

    _EVALUATORS: dict[Enum, Callable[..., BaseEvaluator]] = {}

    @classmethod
    def _register_defaults(cls):
        """Register default evaluators."""
        if not cls._EVALUATORS:
            try:
                from app.core.stages.evaluation.evaluators.feature_selection_evaluator import (
                    FeatureSelectionEvaluator,
                )
                from app.core.stages.evaluation.evaluators.model_selection_evaluator import (
                    ModelSelectionEvaluator,
                )
                from app.core.stages.evaluation.evaluators.data_handler_evaluator import (
                    DataHandlerEvaluator,
                )

                cls._EVALUATORS = {
                    Stages.DATA_HANDLER: DataHandlerEvaluator,
                    Stages.FEATURE_SELECTION: FeatureSelectionEvaluator,
                    Stages.MODEL_SELECTION: ModelSelectionEvaluator,
                    Stages.FINE_TUNING: FineTuningEvaluator,
                    Stages.MODEL_ENSEMBLE: ModelEnsembleEvaluator,
                }
                logger.debug("Default evaluators registered")
            except ImportError as e:
                logger.error(f"Error registering: {e}")

    @classmethod
    def create(cls, stage, context):
        """Create an evaluator for the given stage.

        Returns None if no evaluator is registered for the stage,
        allowing callers to handle missing evaluators gracefully.
        """
        cls._register_defaults()
        try:
            evaluator_class = cls._EVALUATORS.get(stage)
        except ValueError as e:
            raise ValueError(f"Invalid stage: {stage}") from e
        if not evaluator_class:
            logger.warning(
                f"No evaluator registered for stage: {stage}. "
                f"Skipping evaluation for this stage."
            )
            return None
        return evaluator_class(stage=stage, context=context)

    @classmethod
    def register(cls, stage, evaluator_class):
        """Register a new evaluator for a stage."""
        cls._EVALUATORS[stage] = evaluator_class
        logger.info(f"Registered: {evaluator_class.__name__}")
