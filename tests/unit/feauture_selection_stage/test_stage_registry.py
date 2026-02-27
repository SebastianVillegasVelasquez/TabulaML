import pytest

from app.core.context.stages import Stages
import app.core.stages.registry


@pytest.mark.parametrize("stage, expected_experiments", [
    (Stages.FEATURE_SELECTION, app.core.stages.registry._STAGE_REGISTRY[Stages.FEATURE_SELECTION]),
    (Stages.MODEL_SELECTION, app.core.stages.registry._STAGE_REGISTRY[Stages.MODEL_SELECTION])
])
def test_get_stage_experiments_returns_correct_stage(stage, expected_experiments):
    experiments = app.core.stages.registry.get_stage_experiments(stage)
    assert experiments == expected_experiments

def test_get_stage_experiments_raises_for_invalid_stage():
    with pytest.raises(ValueError):
        app.core.stages.registry.get_stage_experiments("Not a stage")