from typing import Dict, Any, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.context import Context
from app.core.enums import EvaluationType
from app.core.enums import Stages
from app.core.experiments.experiment_result import ExperimentResult
from app.core.stages.data_inspection.pipeline_builder import PipelineBuilder
from app.core.stages.super_classes.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from app.utils.logger import logger


class Experiment:
    """Generic experiment runner for machine learning pipeline evaluation.

    This class is intentionally kept flexible so it can be reused across
    different ML workflow stages:

    - Feature selection
    - Model selection
    - Hyperparameter tuning
    - Feature engineering experiments

    It does **not** decide which model or experiment configuration is
    better; it only executes the pipeline, collects metrics, and
    (optionally) extracts the selected features.

    Attributes:
        name: Human-readable identifier for this experiment run.
        pipeline: Builder object responsible for constructing the
            :class:`sklearn.pipeline.Pipeline` on demand.
        cv: Number of cross-validation folds.
        stage: The workflow stage this experiment belongs to.  Controls
            whether feature extraction is attempted after evaluation.
        context: Shared runtime context (data splits, task model_based, etc.).
        metadata: Arbitrary key-value pairs attached to this run for
            logging or downstream consumption.
        evaluation_type: Strategy used to score the pipeline
            (e.g. default cross-validation, threshold-based, etc.).
        threshold: Optional decision threshold forwarded to the
            evaluation strategy.
        selected_features: List of feature name strings chosen by the
            selector step.  Populated only when ``stage`` is
            :attr:`~app.core.enums.Stages.FEATURE_SELECTION`;
            ``None`` otherwise.
    """

    def __init__(
        self,
        name: str,
        pipeline: PipelineBuilder,
        context: Context,
        stage: Stages,
        metadata: Dict[str, Any],
        cv: int = 5,
        threshold: Optional[float] = None,
        evaluation_type: EvaluationType = EvaluationType.DEFAULT,
    ) -> None:
        """Initialises the experiment without running it.

        Args:
            name: Human-readable label for this experiment (e.g.
                ``"lasso_feature_selection"``).
            pipeline: A :class:`PipelineBuilder` whose
                :meth:`~PipelineBuilder.build` method returns a fresh
                :class:`sklearn.pipeline.Pipeline` each time it is called.
            context: Runtime context shared across the experiment suite
                (data splits, target encoder, task metadata, etc.).
            stage: The :class:`~app.core.enums.Stages` constant that
                describes where this experiment sits in the ML workflow.
                When set to :attr:`~app.core.enums.Stages.FEATURE_SELECTION`
                the runner will call :meth:`_extract_features` after
                evaluation and populate :attr:`selected_features`.
            cv: Number of stratified cross-validation folds.
                Defaults to 5.
            metadata: Optional free-form dictionary stored alongside the
                result (e.g. hyperparameter grid search coordinates).
                Defaults to an empty dict.
            threshold: Optional decision threshold forwarded verbatim to
                the evaluation strategy.  When ``None`` the strategy uses
                its own default.
            evaluation_type: :class:`~app.core.enums.EvaluationType`
                constant that selects the concrete evaluation strategy
                via :class:`~app.core.stages.super_classes.evaluation_strategy.evaluation_factory.EvaluationFactory`.
                Defaults to :attr:`~app.core.enums.EvaluationType.DEFAULT`.
        """
        self.selected_features = None
        self.name = name
        self.pipeline = pipeline
        self.cv = cv
        self.stage = stage
        self.context = context
        self.metadata = metadata
        self.evaluation_type = evaluation_type
        self.threshold = threshold

    def run(self, X: pd.DataFrame, y: pd.Series) -> ExperimentResult:
        """Executes the experiment and returns an :class:`ExperimentResult`.

        Steps performed:

        1. Build the sklearn pipeline via :attr:`pipeline`.
        2. Select the evaluation strategy and run cross-validated scoring.
        3. If :attr:`stage` is
           :attr:`~app.core.enums.Stages.FEATURE_SELECTION`, fit the
           pipeline on the full dataset and extract the names of the
           features chosen by the selector step.
        4. Package everything into an :class:`ExperimentResult` and return.

        Args:
            X: Input feature matrix.  Column names must be strings so
                that selected feature names can be propagated correctly.
            y: Target label series aligned with ``X``.

        Returns:
            ExperimentResult: Container holding the pipeline, aggregated
            cross-validation metrics, run configuration, and (when
            applicable) the list of selected feature names.
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

        if self.stage == Stages.FEATURE_SELECTION:
            self.selected_features = self._extract_features(pipeline, X, y)

        experiment_result = ExperimentResult(
            name=self.name,
            pipeline=pipeline,
            metrics=mean_metrics,
            config={"cv": self.cv},
            metadata=self.metadata,
            selected_features=self.selected_features,
        )

        logger.info(
            f"Experiment {experiment_result.name} finished with features: {experiment_result.selected_features}"
        )

        return experiment_result

    def _get_evaluation_type(self) -> EvaluationStrategy:
        """Resolves the concrete evaluation strategy for this experiment.

        Returns:
            EvaluationStrategy: The strategy instance produced by
            :class:`~app.core.stages.super_classes.evaluation_strategy.evaluation_factory.EvaluationFactory`
            for :attr:`evaluation_type`.
        """
        from app.core.stages.super_classes.evaluation_strategy.evaluation_factory import (
            EvaluationFactory,
        )

        return EvaluationFactory.create(self.evaluation_type)

    @staticmethod
    def _extract_features(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Fits the pipeline and extracts the names of the selected features.

        Supports two selector protocols:

        - **sklearn selectors** — any step that exposes a
          ``get_support()`` method (e.g. :class:`~sklearn.feature_selection.SelectKBest`,
          :class:`~sklearn.feature_selection.RFE`).
        - **ShapSelector** — the custom selector that stores chosen
          column positions in a ``selected_idx_`` attribute.

        The selector is assumed to be the second-to-last step in the
        pipeline (or the first step when the pipeline has only two
        steps).  All earlier steps are treated as preprocessors and are
        used to map transformed column positions back to the **original**
        DataFrame column names supplied via ``X``.

        .. important::
            ``X`` **must** be a :class:`pandas.DataFrame` with string
            column names.  Passing a plain numpy array will cause the
            method to fall back to positional string labels
            (``"0"``, ``"1"``, …) rather than meaningful feature names.

        Args:
            pipeline: A **fitted** (or about to be fitted)
                :class:`~sklearn.pipeline.Pipeline`.  The method calls
                ``pipeline.fit(X, y)`` internally, so do **not** pass an
                already-fitted instance if refitting would be harmful.
            X: Input feature matrix as a :class:`pandas.DataFrame`.
                Column names are used to reconstruct the names of the
                selected features after preprocessing.
            y: Target label series aligned with "X".

        Returns:
            list[str]: Ordered list of selected feature name strings,
            guaranteed to be usable as "df[selected_features]".  The
            list is never empty when the selector returns at least one
            feature; if the selector model_based is unrecognized, all column
            names are returned with a warning.
        """
        logger.debug(
            f"Starting _extract_features with pipeline steps: {[name for name, _ in pipeline.steps]}"
        )

        steps = pipeline.steps
        pipeline.fit(X, y)

        selector_index = 0 if len(steps) == 2 else len(steps) - 2
        selector = steps[selector_index][1]

        logger.debug(f"Selector model_based: {type(selector).__name__}")
        logger.debug(f"Selector index: {selector_index}, Total steps: {len(steps)}")

        # Capture original column names *before* any transformation so
        # that positional indices from the selector can be mapped back to
        # meaningful string names.
        original_feature_cols: list[str] = list(X.columns.astype(str))

        if selector_index > 0:
            preprocessor = Pipeline(steps[:selector_index])
            X_transformed = preprocessor.fit_transform(X, y)

            # Try to get feature names from the preprocessor using sklearn's
            # get_feature_names_out() method (available in sklearn >= 1.0).
            # This handles transformers like OneHotEncoder correctly.
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    feature_cols = list(preprocessor.get_feature_names_out(original_feature_cols))
                    logger.debug(
                        f"Retrieved {len(feature_cols)} feature names from preprocessor via get_feature_names_out()"
                    )
                except Exception as e:
                    # Fallback if get_feature_names_out() fails
                    logger.debug(
                        f"get_feature_names_out() failed: {e}; falling back to positional labels"
                    )
                    feature_cols = [str(i) for i in range(X_transformed.shape[1])]
            else:
                # Fallback for older sklearn versions
                logger.debug(
                    "Preprocessor does not have get_feature_names_out(); using positional labels"
                )
                feature_cols = [str(i) for i in range(X_transformed.shape[1])]

            logger.debug(f"Features after preprocessing: {len(feature_cols)}")
        else:
            X_transformed = X
            feature_cols = original_feature_cols
            logger.debug(f"No preprocessing, using original features: {len(feature_cols)}")

        # Resolve selected feature names according to the selector protocol.
        if hasattr(selector, "get_support"):
            # Standard sklearn selector protocol.
            support_mask = selector.get_support()
            selected_feature_names = [
                feature_cols[i] for i, is_selected in enumerate(support_mask) if is_selected
            ]

        elif hasattr(selector, "selected_idx_"):
            # ShapSelector protocol: selected_idx_ holds integer positions.
            selected_indices = selector.selected_idx_
            selected_feature_names = [feature_cols[int(i)] for i in selected_indices]

        else:
            selected_feature_names = feature_cols
            logger.warning(
                f"Unknown selector model_based: {type(selector).__name__}; returning all features."
            )

        # Final guarantee: every element in the returned list is a plain
        # Python str, regardless of how column names were sourced.
        selected_feature_names = [str(f) for f in selected_feature_names]

        return selected_feature_names
