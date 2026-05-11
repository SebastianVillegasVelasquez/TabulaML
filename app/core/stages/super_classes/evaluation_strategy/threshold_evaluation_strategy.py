from app.core.stages.super_classes.evaluation_strategy.evaluation_strategy import (
    EvaluationStrategy,
)
from app.utils.logger import logger


class ThresholdEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, pipeline, X, y, context, cv=5, threshold=None):
        """Evaluate a model using probability thresholding.

        This strategy performs cross-validated probability predictions and applies
        a custom threshold to convert probabilities into class labels.

        Args:
            pipeline: Pipeline or estimator.
            X: Features.
            y: Target labels.
            context: Run context with scoring configuration.
            cv (int): Number of folds.
            threshold (float): Decision threshold for classification.

        Returns:
            dict: Dictionary with computed evaluation metrics.
        """
        from sklearn.model_selection import cross_val_predict

        y_prob = cross_val_predict(pipeline, X, y, method="predict_proba", cv=cv)

        y_prob_pos = y_prob[:, 1]  # Assuming binary classification

        y_pred = (y_prob_pos >= threshold).astype(int)

        metrics = self._build_metrics(self, y_pred, y)
        logger.info(f"Metrics: {metrics}")

        return self._build_metrics(self, y_pred, y)

    @staticmethod
    def _build_metrics(self, y_pred, y_train):
        """
        Calculate evaluation metrics based on predicted labels and true labels.

        Args:
            y_pred: Predicted class labels after applying a threshold.
            y_train: True class labels.

        Returns:
            dict: Dictionary containing evaluation metrics such as accuracy, F1 score, ROC AUC, precision, and recall.

        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            roc_auc_score,
            precision_score,
            recall_score,
        )

        return {
            "accuracy": accuracy_score(y_train, y_pred),
            "f1": f1_score(y_train, y_pred, average="macro"),
            "roc_auc": roc_auc_score(y_train, y_pred),
            "precision": precision_score(y_train, y_pred, average="macro"),
            "recall": recall_score(y_train, y_pred, average="macro"),
        }
