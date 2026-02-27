from unittest.mock import MagicMock

import pandas as pd
import pytest
from app.core.context.stages import Stages
from app.core.stages.feature_selection.feature_selection_stage import FeatureSelectionStage


# @pytest.fixture
# def mock_context():
#     context = MagicMock()
#     context.stage_results = {
#         Stages.DATA_HANDLER: {
#             "results": {
#                 "feature_configs": [
#                     {"name": "age", "feature_type": "numerical", "encoding": None},
#                     {"name": "gender", "feature_type": "categorical", "encoding": "onehot"},
#                 ]
#             }
#         }
#     }
#     context.current_stage = Stages.FEATURE_SELECTION
#     context.config.X_train = pd.DataFrame({
#         "age": [25, 30, 35],
#         "gender": ["M", "F", "M"]
#     })
#     context.config.y_train = pd.Series([0, 1, 0])
#     return context
#
#
# @pytest.fixture
# def mock_get_stage_experiments(monkeypatch):
#     mock = MagicMock()
#     mock.return_value = [
#         MagicMock(
#             name="experiment1",
#             builder=lambda preprocessing: MagicMock(build=lambda: MagicMock()),
#             metadata={"type": "dummy"}
#         ),
#         MagicMock(
#             name="experiment2",
#             builder=lambda preprocessing: MagicMock(build=lambda: MagicMock()),
#             metadata={"type": "dummy"}
#         ),
#     ]
#     monkeypatch.setattr(
#         "app.core.stages.feature_selection.feature_selection_stage.get_stage_experiments",
#         mock
#     )
#     return mock
#
#
# @pytest.fixture
# def mock_logger(monkeypatch):
#     mock = MagicMock()
#     monkeypatch.setattr("app.core.stages.feature_selection.feature_selection_stage.logger", mock)
#     return mock


# def test_feature_selection_stage_run(mock_context, mock_get_stage_experiments, mock_logger):
#     stage = FeatureSelectionStage(context=mock_context)
#
#     results = stage.run()
#
#     assert len(results) == 2  # Ensure two experiments are processed
#     mock_get_stage_experiments.assert_called_once_with(Stages.FEATURE_SELECTION)
#     mock_logger.info.assert_any_call("Running feature selection stage...")
#     mock_logger.info.assert_any_call("Finished experiment FEATURE_SELECTION_experiment1")
#     mock_logger.info.assert_any_call("Finished experiment FEATURE_SELECTION_experiment2")
