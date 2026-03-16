from app.core.context.stages import Stages
from app.utils.logger import logger


class EvaluatorFactory:
    """Factory for creating stage-specific evaluators."""
    
    _EVALUATORS = {}
    
    @classmethod
    def _register_defaults(cls):
        """Register default evaluators."""
        if not cls._EVALUATORS:
            try:
                from app.core.stages.evaluation.feature_selection_evaluator import FeatureSelectionEvaluator
                from app.core.stages.evaluation.model_selection_evaluator import ModelSelectionEvaluator
                cls._EVALUATORS = {
                    Stages.FEATURE_SELECTION: FeatureSelectionEvaluator,
                    Stages.MODEL_SELECTION: ModelSelectionEvaluator,
                }
                logger.debug("Default evaluators registered")
            except ImportError as e:
                logger.error(f"Error registering: {e}")
    
    @classmethod
    def create(cls, stage, context):
        """Create evaluator for the given stage."""
        cls._register_defaults()
        evaluator_class = cls._EVALUATORS.get(stage)
        if not evaluator_class:
            raise ValueError(f"No evaluator for stage: {stage}")
        return evaluator_class(stage=stage, context=context)
    
    @classmethod
    def register(cls, stage, evaluator_class):
        """Register a new evaluator for a stage."""
        cls._EVALUATORS[stage] = evaluator_class
        logger.info(f"Registered: {evaluator_class.__name__}")

