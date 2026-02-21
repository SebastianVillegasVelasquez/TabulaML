from typing import Tuple

import pandas as pd

from app.core.context.metrics import DEFAULT_METRICS
from app.core.context.problems_type import ProblemsType
from app.core.context.project_config import ProjectConfig
from app.core.context.run_context import RunContext


def _decouple_tuples(X, y):
    X_train, y_train = X
    X_test, y_test = y
    return X_train, y_train, X_test, y_test


def init_context(problem_type: ProblemsType = ProblemsType.CLASSIFICATION,
                 X: Tuple[pd.DataFrame, pd.Series] = None,
                 y: Tuple[pd.DataFrame, pd.Series] = None) -> RunContext:

    X_train, y_train, X_test, y_test = _decouple_tuples(X, y)

    return (RunContext
        (
        ProjectConfig
        (problem_type=problem_type,
         X_train=X_train,
         y_train=y_train,
         X_test=X_test,
         y_test=y_test,
         scoring=DEFAULT_METRICS[problem_type],
         random_state=42
         ))
    )
