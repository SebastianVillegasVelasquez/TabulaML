from typing import List

from app.core.context.run_context import RunContext
from app.core.context.stages import Stages
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.stages.feature_selection.feature_selection_experiments import FEATURE_SELECTION_EXPERIMENTS
from app.core.stages.model_selection.model_selection_experiments import get_model_selection_experiments
from app.utils.logger import logger


def get_stage_experiments(stage: Stages,
                          context: RunContext = None) -> List[ExperimentDefinition]:
    """
    Returns all experiment definitions registered for a given stage.
    """

    if stage == Stages.FEATURE_SELECTION:
        return FEATURE_SELECTION_EXPERIMENTS
    elif stage == Stages.MODEL_SELECTION:
        return get_model_selection_experiments(context=context)
    else:
        logger.debug(f"Stage '{stage}' not registered.")
        raise ValueError(f"Stage '{stage}' not registered.")
