import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40]
    })
    y_train = pd.Series([0, 1, 0, 1])

    X_test = pd.DataFrame({
        "feature1": [5, 6],
        "feature2": [50, 60]
    })
    y_test = pd.Series([1, 0])

    return (X_train, y_train), (X_test, y_test)


from app.core.context import init_context
from app.core.enums import ProblemsType


@pytest.fixture(
    params=[
        ProblemsType.CLASSIFICATION,
        ProblemsType.REGRESSION
    ]
)
def run_context_params(request, sample_data):
    X, y = sample_data

    return init_context(
        problem_type=request.param,
        X=X,
        y=y
    )

# @pytest.fixture
# def run_context(sample_data):
#     X, y = sample_data
#
#     return init_context(
#         problem_type=request,
#         X=X,
#         y=y
#     )
