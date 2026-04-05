from app.core.stages.super_classes.evaluation_strategy.evaluation_strategy import EvaluationStrategy


class DefaultEvaluationStrategy(EvaluationStrategy):

    def evaluate(self, pipeline, X, y, context, cv=5, threshold=None):
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
            error_score="raise"
        )

        mean_metrics = {}

        for metric_name, values in scores.items():
            if metric_name.startswith("train_") or metric_name.startswith("test_"):

                mean_value = np.mean(values)

                if metric_name.endswith(("neg_mean_squared_error",
                                         "neg_mean_absolute_error")):
                    mean_value = -mean_value

                mean_metrics[metric_name] = mean_value

        return mean_metrics

