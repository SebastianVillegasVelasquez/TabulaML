from typing import List

from app.core.context.context import Context
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.fine_tuning.fine_tuning_experiments import get_fine_tuning_experiments


class FineTuningExperimentsFactory(BaseExperimentFactory):

    def create_experiments(self, context: Context = None) -> List[ExperimentDefinition]:
        if context is None:
            raise ValueError(f"Context parameter must not be None for FineTuningExperimentsFactory")

        return get_fine_tuning_experiments(context=context)
