import sys
from pathlib import Path

import pandas as pd
import pytest

from app.core.context.init_context import init_context
from app.core.enums import ProblemType, ModelRetrieveType
from app.core.model_bank import ModelRetrieveFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    y_train = pd.Series([0, 1, 0, 1])

    X_test = pd.DataFrame({"feature1": [5, 6], "feature2": [50, 60]})
    y_test = pd.Series([1, 0])

    return (X_train, y_train), (X_test, y_test)


@pytest.fixture(params=[ProblemType.CLASSIFICATION, ProblemType.REGRESSION])
def run_context(request, sample_data):
    X, y = sample_data

    return init_context(problem_type=request.param, X=X, y=y)


@pytest.fixture
def retrieve_models():
    return (
        ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.SELECTOR, problem_type=ProblemType.CLASSIFICATION
        )
    ).load_defaults()


@pytest.fixture
def retrieve_selectors():
    return (
        ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.BASELINE, problem_type=ProblemType.CLASSIFICATION
        )
    ).load_defaults()
