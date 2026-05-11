from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.context import Context


class EvaluationStrategy(ABC):
    """
    Abstract base class for evaluation strategies.
    """

    @abstractmethod
    def evaluate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        context: Context,
        return_estimator: bool = False,
        cv: int = 5,
        threshold: float | None = None,
    ):
        pass
