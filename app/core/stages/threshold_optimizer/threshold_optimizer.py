import numpy as np

from app.core.context.run_context import RunContext


class ThresholdOptimizer():

    def __init__(self, context: RunContext):
        self.context = context

    def find_best_threshold(self, model):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        if not self._is_the_model_probabilistic(model):
            raise ValueError("The model is not probabilistic")

        X = self.context.config.X_train
        y = self.context.config.y_train

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model.fit(X, y)

        y_proba = model.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.0, 1.0, 100)

        best_threshold = 0.5
        best_score = -1

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            score = f1_score(y_val, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    @staticmethod
    def _is_the_model_probabilistic(model):
        return hasattr(model, 'predict_proba') and callable(model.predict_proba)

    # def _get_strategy_for_threshold_finding(self, metric: str):
    #     if metric == "f1":
    #         return self._find_threshold_by_f1
    #     else:
    #         raise ValueError(f"Unsupported metric for threshold finding: {metric}")