from typing import Dict, Any, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.context import Context
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.enums import EvaluationType
from app.core.stages.super_classes.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from app.utils.logger import logger


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
        pipeline: Pipeline,
        context: Context,
        cv: int = 5,
        metadata: Dict[str, Any] | None = None,
        threshold: Optional[float] = None,
        evaluation_type: EvaluationType = EvaluationType.DEFAULT,
    ):
        """
        :param name: Unique experiment name.
        :param pipeline_builder: Either a PipelineBuilder or a pre-constructed sklearn Pipeline/estimator.
                                If PipelineBuilder, it will be built to create a Pipeline.
                                If Pipeline or estimator, it will be used directly.
        :param cv: Number of cross-validation folds.
        :param metadata: Optional experiment configuration for tracking.
        :param evaluation_type: Type of evaluation to perform.
        """
        self.name = name
        self.pipeline = pipeline
        self.cv = cv
        self.context = context
        self.metadata = metadata or {}
        self.evaluation_type = evaluation_type
        self.threshold = threshold

    def run(self, X: pd.DataFrame, y: pd.Series) -> ExperimentResult:
        """
        Executes cross-validation and fits the final model.

        :param X: Feature dataframe.
        :param y: Target series.
        :return: ExperimentResult object.
        """
        evaluation = self._get_evaluation_type()

        mean_metrics = evaluation.evaluate(
            self.pipeline, X, y, self.context, self.cv, threshold=self.threshold
        )

        logger.debug(f"Mean metrics: {mean_metrics} using {self.evaluation_type}")

        experiment_result = ExperimentResult(
            name=self.name,
            pipeline=self.pipeline,
            metrics=mean_metrics,
            config={"cv": self.cv, "scoring": self.context.config.scoring, **self.metadata},
        )

        return experiment_result

    def _get_evaluation_type(self) -> EvaluationStrategy:
        from app.core.stages.super_classes.evaluation_strategy.evaluation_factory import (
            EvaluationFactory,
        )

        return EvaluationFactory.create(self.evaluation_type)
