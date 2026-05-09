from typing import List

from app.core.context.context import Context
from app.core.enums import Stages
from app.core.experiments import ExperimentDefinition
from app.core.stages.factories.stage_experiment_factory import StageExperimentFactory


def get_stage_experiments(stage: Stages, context: Context = None) -> List[ExperimentDefinition]:
    """
    It communicates with the Experiment factory and passes a certain stage
    to retrieve the factory for that stage.
    Then the factory calls the "create_experiments" method to get the experiments.

    Args:
        stage(Stages): The stage for which the experiments are to be retrieved.
        context(Context): The context object containing the pipeline configuration and state.
    Return:
        List[ExperimentDefinition]: A list of ExperimentDefinition objects for the specified stage.
    """
    factory = StageExperimentFactory.create(stage)
    return factory.create_experiments(context)
