from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from app.core.context.run_context import RunContext
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.ml.pipeline_builder import PipelineBuilder


class Experiment:
    """
    Generic experiment runner.

    This class is intentionally kept flexible so it can be reused for:
    - Feature selection
    - Model selection
    - Hyperparameter tuning
    - Feature engineering experiments

    It does NOT decide which model to use.
    It does NOT decide which experiment is better.
    It only executes and returns results.
    """

    def __init__(
            self,
            name: str,
            pipeline_builder: PipelineBuilder,
            context: RunContext,
            cv: int = 5,
            metadata: Dict[str, Any] | None = None
    ):
        """
        :param name: Unique experiment name.
        :param pipeline_builder: Responsible for constructing the sklearn pipeline.
        :param cv: Number of cross-validation folds.
        :param metadata: Optional experiment configuration for tracking.
        """
        self.name = name
        self.pipeline_builder = pipeline_builder
        self.cv = cv
        self.context = context
        self.metadata = metadata or {}

    def run(self,
            X: pd.DataFrame,
            y: pd.Series) -> ExperimentResult:
        """
        Executes cross-validation and fits the final model.

        :param X: Feature dataframe.
        :param y: Target series.
        :return: ExperimentResult object.
        """
        if isinstance(self.pipeline_builder, PipelineBuilder):
            pipeline = self.pipeline_builder.build()
        else:
            pipeline = self.pipeline_builder


        scores = cross_validate(
            pipeline,
            X,
            y,
            scoring=self.context.config.scoring,
            cv=self.cv,
            n_jobs=-1,
            return_train_score=True,
            error_score="raise"
        )

        # Aggregate metrics properly
        mean_metrics = {}

        for metric_name, values in scores.items():

            if metric_name.startswith("train_") or metric_name.startswith("test_"):

                mean_value = (np.mean(values))

                # Fix the sklearn convention for MSE and MAE
                if metric_name.endswith(("neg_mean_squared_error",
                                         "neg_mean_absolute_error")):
                    mean_value = -mean_value

                mean_metrics[metric_name] = mean_value

        # Fit the final pipeline on full dataset (artifact ready)
        pipeline.fit(X, y)

        experiment_result = ExperimentResult(
            name=self.name,
            pipeline=pipeline,
            metrics=mean_metrics,
            config={
                "cv": self.cv,
                "scoring": self.context.config.scoring,
                **self.metadata
            }
        )

        return experiment_result
