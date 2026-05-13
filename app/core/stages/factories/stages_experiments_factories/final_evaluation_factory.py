from typing import List

from app.core.context import Context
from app.core.experiments import ExperimentDefinition
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.final_evaluation.final_evaluation_experiments import (
    get_final_evaluation_experiments,
)


class FinalEvaluationFactory(BaseExperimentFactory):
    def create_experiments(
        self, context: Context | None = None
    ) -> List[ExperimentDefinition]:
        if context is None:
            raise ValueError(
                "Context parameter must not be None for ModelEnsembleExperimentFactory\n"
                "It has to be able to get the models from the feature selection stage"
            )

        return get_final_evaluation_experiments(context=context)
