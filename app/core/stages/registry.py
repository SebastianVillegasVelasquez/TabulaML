from typing import List

from app.core.context.stages import Stages
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.stages.feature_selection.feature_selection import FEATURE_SELECTION_EXPERIMENTS
from app.core.domain.experiments.model_selection import MODEL_SELECTION_EXPERIMENTS

_STAGE_REGISTRY = {
    Stages.FEATURE_SELECTION: FEATURE_SELECTION_EXPERIMENTS,
    Stages.MODEL_SELECTION: MODEL_SELECTION_EXPERIMENTS,
}


def get_stage_experiments(stage: Stages) -> List[ExperimentDefinition]:
    """
    Returns all experiment definitions registered for a given stage.
    """

    if stage not in _STAGE_REGISTRY:
        raise ValueError(f"Stage '{stage}' not registered.")

    return _STAGE_REGISTRY[stage]
