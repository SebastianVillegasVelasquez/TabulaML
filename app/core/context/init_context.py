from typing import Tuple, List

import pandas as pd

from app.core.metrics.metrics import DEFAULT_METRICS
from app.core.enums.problems_type import ProblemsType
from app.core.context.run_context import ProjectConfig
from app.core.context.run_context import RunContext


def init_context(problem_type: ProblemsType = ProblemsType.CLASSIFICATION,
                 X: Tuple[pd.DataFrame, pd.Series] = None,
                 y: Tuple[pd.DataFrame, pd.Series] = None,
                 priority_metric: str = None) -> RunContext:

    if X is None or y is None:
        raise ValueError("X and y must not be None")

    if problem_type not in [ProblemsType.CLASSIFICATION, ProblemsType.REGRESSION]:
        raise ValueError(f"Invalid problem type: {problem_type}")

    X_train, y_train, X_test, y_test = _decouple_tuples(X, y)


    context = RunContext()

    context.config = ProjectConfig(
        problem_type,
        X_train,
        y_train,
        X_test,
        y_test,
        DEFAULT_METRICS[problem_type],
        random_state=42,
        priority_metric=_get_priority_metric(problem_type, priority_metric),
        priority_metric_normalized = priority_metric
    )
    context.metadata = _get_metadata(X_train)

    return context


def _get_metadata(X):
    return {
        "total_columns": len(X.columns),
        "total_rows": len(X),
        "original_shape": X.shape
    }


def _decouple_tuples(X, y):
    if len(X) != 2 or len(y) != 2:
        raise ValueError("X and y must be tuples of (train, test)")

    X_train, y_train = X
    X_test, y_test = y
    return X_train, y_train, X_test, y_test


def _get_priority_metric(problem_type, priority_metric=None):
    if priority_metric is not None:
        return f"test_{priority_metric}"
    return "test_f1" if problem_type == ProblemsType.CLASSIFICATION else "test_neg_mean_squared_error"
