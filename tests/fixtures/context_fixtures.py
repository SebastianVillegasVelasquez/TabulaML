import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock

from app.core.context.context import Context, ProjectConfig
from app.core.enums import ProblemType
from app.services.loader import DatasetBundle


@pytest.fixture
def mock_project_config():
    config = Mock(spec=ProjectConfig)
    config.priority_metric = "test_f1"
    config.scoring = ["f1"]
    config.problem_type = ProblemType.CLASSIFICATION
    config.random_state = 42
    return config


@pytest.fixture
def mock_context(mock_project_config):
    context = Mock(spec=Context)
    context.config = mock_project_config
    context.stage_results = {}
    context.update_stage_context = MagicMock()
    return context


@pytest.fixture
def real_dataset_bundle():
    X_train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0])
    X_test = pd.DataFrame({"f1": [6.0, 7.0], "f2": [60.0, 70.0]})
    y_test = pd.Series([1, 0])
    return DatasetBundle(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


@pytest.fixture
def real_classification_context(real_dataset_bundle):
    return Context.create(
        dataset=real_dataset_bundle,
        problem_type=ProblemType.CLASSIFICATION,
        target_column="target",
    )
