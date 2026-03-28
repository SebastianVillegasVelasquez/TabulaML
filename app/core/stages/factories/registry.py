from typing import List

from app.core.context.run_context import RunContext
from app.core.enums.stages import Stages
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.stages.factories.stage_experiment_factory import StageExperimentFactory


def get_stage_experiments(stage: Stages,
                          context: RunContext = None) -> List[ExperimentDefinition]:
    """
    Returns all experiment definitions registered for a given stage.
    """
    factory = StageExperimentFactory.create(stage, context)
    return factory.create_experiments(context)