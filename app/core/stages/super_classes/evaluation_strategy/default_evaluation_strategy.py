import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.context import Context
from app.core.stages.super_classes.evaluation_strategy.evaluation_strategy import (
    EvaluationStrategy,
)


class DefaultEvaluationStrategy(EvaluationStrategy):
    """
    Performs a default evaluation using a cross_validate,
    this strategy is used by stages as
    1. Feature Selection
    2. Model Selection
    3. Fine Tuning

    It is not used to Theshold evaluation, it needs a special evaluation strategy

    """

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
        from sklearn.model_selection import cross_validate
        import numpy as np

        scores = cross_validate(
            pipeline,
            X,
            y,
            scoring=context.config.scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            return_estimator=return_estimator,
            error_score="raise",
        )

        mean_metrics = {}

        for metric_name, values in scores.items():
            if metric_name.startswith("train_") or metric_name.startswith("test_"):
                mean_value = np.mean(values)

                if metric_name.endswith(
                    ("neg_mean_squared_error", "neg_mean_absolute_error")
                ):
                    mean_value = -mean_value

                mean_metrics[metric_name] = mean_value

        return mean_metrics
