import numpy as np
from sklearn.base import BaseEstimator

from app.core.context import RunContext
from app.utils.logger import logger


class ThresholdOptimizer:
    """
    This class is responsible for optimizing the threshold of a classifier.
    Should only work for classification problems.

    Responsibilities:
    - Evaluates the model's predicted probabilities on a validation set
    - Selects the threshold that maximizes the specified metric (e.g., F1-score, precision, recall)
    - Returns the best threshold and corresponding metric score for deployment
    """

    def __init__(self, context: RunContext):
        self.context = context

    def find_best_threshold(self, model: BaseEstimator) -> dict:
        """
        Finds the optimal decision threshold for a probabilistic classification model
        based on a specified evaluation metric.

        This method splits the training data into a training and validation subset,
        fits the model, and evaluates multiple threshold values on the validation set
        to determine which threshold maximizes the selected metric.

        Args:
            model: A trained or untrained classification model. The model must implement
                the `predict_proba` method to be considered probabilistic.

        Returns:
            dict: A dictionary containing:
                - 'best_threshold' (float): The threshold value that maximizes the metric.
                - 'best_score' (float): The best metric score achieved.

            If the model is not probabilistic, the original model is returned.

        Raises:
            ValueError: If the configured evaluation metric is not supported.

        Notes:
            - The metric used for evaluation is defined in
              `self.context.config.priority_metric_normalized`.
            - The dataset is split using an 80/20 train-validation split with stratification.
            - Thresholds are evaluated in the range [0.0, 1.0] using 100 evenly spaced values.
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        if not self._is_the_model_probabilistic(model):
            logger.info("Model is not probabilistic, cannot optimize threshold.")
            return model

        X = self.context.config.X_train
        y = self.context.config.y_train

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Starting threshold optimization...")
        model.fit(X, y)

        y_proba = model.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.0, 1.0, 100)

        best_threshold = 0.5
        best_score = -1

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            score = self._get_score(
                context=self.context,
                y_val=y_val,
                y_pred=y_pred
            )

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return {
            'best_threshold': best_threshold,
            'best_score': best_score,
        }

    @staticmethod
    def _is_the_model_probabilistic(model: BaseEstimator) -> bool:
        """
        Checks whether a model supports probability predictions.

        A model is considered probabilistic if it implements a callable
        `predict_proba` method.

        Args:
            model: The model to evaluate.

        Returns:
            bool: True if the model has a callable `predict_proba` method,
            False otherwise.
        """
        return hasattr(model, 'predict_proba') and callable(model.predict_proba)

    @staticmethod
    def _get_score(context: runContext,
                   y_val: np.ndarray,
                   y_pred: np.ndarray) -> float:
        """
        Computes the evaluation score based on the configured metric.

        The metric is selected dynamically from the configuration and
        applied to the validation predictions.

        Args:
            context: An object containing configuration parameters, including
                `priority_metric_normalized`.
            y_val (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.

        Returns:
            float: The computed evaluation score.

        Raises:
            ValueError: If the specified metric is not supported.

        Supported Metrics:
            - "f1": F1 score
            - "precision": Precision score
            - "Recall": Recall score
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        metric = context.config.priority_metric_normalized

        match metric:
            case "f1":
                return f1_score(y_val, y_pred, zero_division=0)
            case "precision":
                return precision_score(y_val, y_pred, zero_division=0)
            case "recall":
                return recall_score(y_val, y_pred, zero_division=0)
            case _:
                raise ValueError(f"Unsupported metric for threshold optimization: {metric}")
