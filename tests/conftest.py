import sys
from pathlib import Path

from data_inspection import DataInspectionStage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from app.core.context import Context, DatasetBundle
from app.core.enums import ProblemType, ModelRetrieveType
from app.core.model_bank import ModelRetrieveFactory


@pytest.fixture
def init_data_inspection_stage(build_context: Context):
    from data_inspection import DataInspectionStage

    stage = DataInspectionStage(build_context)
    yield stage


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    y_train = pd.Series([0, 1, 0, 1])

    X_test = pd.DataFrame({"feature1": [5, 6], "feature2": [50, 60]})
    y_test = pd.Series([1, 0])

    yield (X_train, y_train), (X_test, y_test)


@pytest.fixture
def dataset_bundle(sample_data):
    (X_train, y_train), (X_test, y_test) = sample_data
    yield DatasetBundle(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


@pytest.fixture(params=[ProblemType.CLASSIFICATION, ProblemType.REGRESSION])
def build_context(request, dataset_bundle):
    yield Context.create(
        dataset=dataset_bundle,
        problem_type=request.param,
        target_column="target",
    )


@pytest.fixture
def retrieve_models():
    yield (
        ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.SELECTOR, problem_type=ProblemType.CLASSIFICATION
        )
    ).load_defaults()


@pytest.fixture
def retrieve_selectors():
    yield (
        ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.BASELINE, problem_type=ProblemType.CLASSIFICATION
        )
    ).load_defaults()
