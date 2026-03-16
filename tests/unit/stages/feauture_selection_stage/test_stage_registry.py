from unittest.mock import MagicMock

import pytest

from app.core.context.stages import Stages
import app.core.stages.factories.registry
from app.core.stages.feature_selection.feature_selection_experiments import FEATURE_SELECTION_EXPERIMENTS


@pytest.mark.parametrize("stage, expected_experiments", [
    (Stages.FEATURE_SELECTION, FEATURE_SELECTION_EXPERIMENTS),
    (Stages.MODEL_SELECTION, list)
])
def test_get_stage_experiments_returns_correct_stage(stage, expected_experiments):
    mock = MagicMock()
    experiments = app.core.stages.registry.registry.get_stage_experiments(stage, context=mock)
    assert experiments == expected_experiments

def test_get_stage_experiments_raises_for_invalid_stage():
    with pytest.raises(ValueError):
        app.core.stages.registry.registry.get_stage_experiments("Not a stage")