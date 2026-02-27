from typing import Tuple, List

import pandas as pd

from app.core.context.metrics import DEFAULT_METRICS
from app.core.context.problems_type import ProblemsType
from app.core.context.project_config import ProjectConfig
from app.core.context.run_context import RunContext


def init_context(problem_type: ProblemsType = ProblemsType.CLASSIFICATION,
                 X: Tuple[pd.DataFrame, pd.Series] = None,
                 y: Tuple[pd.DataFrame, pd.Series] = None,
                 priority_metrics: str | List[str] = None) -> RunContext:
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
        priority_metrics=_get_priority_metric(problem_type)
        if priority_metrics is None
        else priority_metrics
    )
    context.metadata = _get_metadata(X_train)

    return context


def _get_metadata(X):
    return {
        "original_columns": f"{list(X.columns)}",
        "total_columns": f"{len(X.columns)}",
        "total_rows": f"{len(X)}",
        "original_shape": f"{X.shape}"
    }


def _decouple_tuples(X, y):
    X_train, y_train = X
    X_test, y_test = y
    return X_train, y_train, X_test, y_test


def _get_priority_metric(problem_type, priority_metric=None):
    if priority_metric is not None:
        return f"test_{priority_metric}"
    return "test_f1" if problem_type == ProblemsType.CLASSIFICATION else "test_neg_mean_squared_error"
