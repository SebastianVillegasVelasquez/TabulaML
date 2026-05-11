from typing import List

from app.core.context.context import Context
from app.core.experiments import ExperimentDefinition
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.model_selection.model_selection_experiments import (
    get_model_selection_experiments,
)


class ModelSelectionExperimentsFactory(BaseExperimentFactory):
    def create_experiments(self, context: Context | None = None) -> List[ExperimentDefinition]:
        if context is None:
            raise ValueError(
                "Context parameter must not be None for ModelSelectionExperimentsFactory\n"
                "It has to be able to get the models from the feature selection stage"
            )

        return get_model_selection_experiments(context=context)
