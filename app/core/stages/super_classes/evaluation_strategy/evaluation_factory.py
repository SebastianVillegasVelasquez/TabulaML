from app.core.enums.evaluations import EvaluationType


class EvaluationFactory:
    _EVALUATIONS = {}

    @classmethod
    def _register_defaults(cls):
        from app.core.stages.super_classes.evaluation_strategy import (
            DefaultEvaluationStrategy,
            ThresholdEvaluationStrategy
        )
        if not cls._EVALUATIONS:
            cls._EVALUATIONS = {
                EvaluationType.DEFAULT: DefaultEvaluationStrategy,
                EvaluationType.THRESHOLD: ThresholdEvaluationStrategy,
            }

    @classmethod
    def create(cls, evaluation_type: EvaluationType):
        cls._register_defaults()
        evaluation_class = cls._EVALUATIONS.get(evaluation_type)
        if not evaluation_class:
            raise ValueError(f"No evaluation for type: {evaluation_type}")
        return evaluation_class()

    @classmethod
    def register(cls, stage, evaluator_class):
        """Register a new evaluator for a stage."""
        cls._EVALUATIONS[stage] = evaluator_class
