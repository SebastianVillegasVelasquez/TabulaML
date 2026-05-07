from typing import Dict, Any, Optional

import pandas as pd
from sklearn import clone
from sklearn.pipeline import Pipeline

from app.core.context import Context
from app.core.enums import EvaluationType
from app.core.stages.data_inspection.pipeline_builder import PipelineBuilder
from app.core.stages.super_classes.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from app.utils.logger import logger
from enums import Stages
from experiments.experiment_result import ExperimentResult


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
            pipeline: PipelineBuilder,
            context: Context,
            stage: Stages,
            cv: int = 5,
            metadata: Dict[str, Any] | None = None,
            threshold: Optional[float] = None,
            evaluation_type: EvaluationType = EvaluationType.DEFAULT,
    ):
        self.name = name
        self.pipeline = pipeline
        self.cv = cv
        self.stage = stage
        self.context = context
        self.metadata = metadata or {}
        self.evaluation_type = evaluation_type
        self.threshold = threshold

    def run(self, X: pd.DataFrame, y: pd.Series) -> ExperimentResult:
        """
        This function handles the execution of the experiment.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.

        Returns:
            ExperimentResult: The experiment result.
        """

        pipeline = self.pipeline.build()

        evaluation = self._get_evaluation_type()

        mean_metrics = evaluation.evaluate(
            pipeline=pipeline,
            X=X,
            y=y,
            context=self.context,
            return_estimator=True,
            cv=self.cv,
            threshold=self.threshold,
        )

        if self.stage is Stages.FEATURE_SELECTION:
            features = self._extract_features(pipeline,
                                              X,
                                              y)

        logger.debug(f"Mean metrics: {mean_metrics} using {self.evaluation_type}")

        experiment_result = ExperimentResult(
            name=self.name,
            pipeline=pipeline,
            metrics=mean_metrics,
            config={"cv": self.cv},
        )

        return experiment_result

    def _get_evaluation_type(self) -> EvaluationStrategy:
        from app.core.stages.super_classes.evaluation_strategy.evaluation_factory import (
            EvaluationFactory,
        )

        return EvaluationFactory.create(self.evaluation_type)

    @staticmethod
    def _extract_features(
            pipeline: Pipeline,
            X,
            y) -> list[str]:
        """
        This method fit the pipeline and extract features selected by the selectors.
        Handles both sklearn selectors (with get_support()) and ShapSelector (with selected_idx_).
        """
        steps = pipeline.steps
        pipeline.fit(X, y)

        selector_index = 0 if len(steps) == 2 else len(steps) - 2
        selector = steps[selector_index][1]

        # Transform X through preprocessing steps before selector
        if selector_index > 0:
            preprocessor = Pipeline(steps[:selector_index])
            X_transformed = preprocessor.transform(X)
            X_transformed = pd.DataFrame(X_transformed)
            feature_cols = X_transformed.columns
        else:
            X_transformed = X
            feature_cols = X.columns

        # Handle both sklearn selectors and ShapSelector
        if hasattr(selector, 'get_support'):
            # Standard sklearn selector with get_support() method
            support_mask = selector.get_support()
            selected_feature_names = list(feature_cols[support_mask])
        elif hasattr(selector, 'selected_idx_'):
            # ShapSelector with selected_idx_ attribute
            selected_indices = selector.selected_idx_
            selected_feature_names = [feature_cols[i] for i in selected_indices]
        else:
            selected_feature_names = list(feature_cols)

        return selected_feature_names


