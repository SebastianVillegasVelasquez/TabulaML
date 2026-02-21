from dataclasses import dataclass
from typing import Union

import pandas as pd

from app.core.context.problems_type import ProblemsType


@dataclass(frozen=True)
class ProjectConfig:
    problem_type: ProblemsType
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scoring: list[str]
    random_state: int
    priority_metrics: Union[str, list[str]] = None
