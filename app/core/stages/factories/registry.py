from typing import List

from app.core.context.context import Context
from app.core.enums import Stages
from experiments import ExperimentDefinition
from app.core.stages.factories.stage_experiment_factory import StageExperimentFactory


def get_stage_experiments(stage: Stages, context: Context = None) -> List[ExperimentDefinition]:
    """
    Returns all experiment definitions registered for a given stage.
    """
    factory = StageExperimentFactory.create(stage, context)
    return factory.create_experiments(context)
