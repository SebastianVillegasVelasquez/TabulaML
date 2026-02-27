import pytest
from sklearn.pipeline import Pipeline

from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.stages.evaluation.evaluator import Evaluator


class TestEvaluator:

    def test_init_with_single_metric(self):
        evaluator = Evaluator(metric="accuracy", mode="max")
        assert evaluator.metric == "accuracy"
        assert evaluator.mode == "max"
        assert evaluator.results == []

    def test_init_with_multiple_metrics(self):
        evaluator = Evaluator(metric=["accuracy", "precision"], mode="max")
        assert evaluator.metric == ["accuracy", "precision"]
        assert evaluator.mode == "max"
        assert evaluator.results == []

    def test_init_with_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be 'max' or 'min'"):
            Evaluator(metric="accuracy", mode="invalid")

    def test_add_result_with_single_metric(self):
        evaluator = Evaluator(metric="accuracy", mode="max")
        result = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.95}
        )
        evaluator.add_result(result)
        assert len(evaluator.results) == 1
        assert evaluator.results[0] == result

    def test_add_result_with_multiple_metrics(self):
        evaluator = Evaluator(metric=["accuracy", "precision"], mode="max")
        result = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.95, "precision": 0.90}
        )
        evaluator.add_result(result)
        assert len(evaluator.results) == 1
        assert evaluator.results[0] == result

    def test_add_result_missing_single_metric(self):
        evaluator = Evaluator(metric="accuracy", mode="max")
        result = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"precision": 0.90}
        )
        with pytest.raises(ValueError, match="Metric 'accuracy' not found in experiment exp1"):
            evaluator.add_result(result)

    def test_add_result_missing_one_of_multiple_metrics(self):
        evaluator = Evaluator(metric=["accuracy", "precision"], mode="max")
        result = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.95}
        )
        with pytest.raises(ValueError, match="Metric 'precision' not found in experiment exp1"):
            evaluator.add_result(result)

    def test_get_best_with_no_results(self):
        evaluator = Evaluator(metric="accuracy", mode="max")
        with pytest.raises(RuntimeError, match="No experiments evaluated."):
            evaluator.get_best()

    def test_get_best_single_metric_max_mode(self):
        evaluator = Evaluator(metric="accuracy", mode="max")

        result1 = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.85}
        )
        result2 = ExperimentResult(
            name="exp2",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.95}
        )
        result3 = ExperimentResult(
            name="exp3",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.90}
        )

        evaluator.add_result(result1)
        evaluator.add_result(result2)
        evaluator.add_result(result3)

        best = evaluator.get_best()
        assert best == result2
        assert best.metrics["accuracy"] == 0.95

    def test_get_best_single_metric_min_mode(self):
        evaluator = Evaluator(metric="mse", mode="min")

        result1 = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"mse": 0.85}
        )
        result2 = ExperimentResult(
            name="exp2",
            pipeline=Pipeline([]),
            metrics={"mse": 0.45}
        )
        result3 = ExperimentResult(
            name="exp3",
            pipeline=Pipeline([]),
            metrics={"mse": 0.60}
        )

        evaluator.add_result(result1)
        evaluator.add_result(result2)
        evaluator.add_result(result3)

        best = evaluator.get_best()
        assert best == result2
        assert best.metrics["mse"] == 0.45

    def test_get_best_multiple_metrics_max_mode(self):
        evaluator = Evaluator(metric=["accuracy", "precision"], mode="max")

        result1 = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.85, "precision": 0.80}
        )
        result2 = ExperimentResult(
            name="exp2",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.85, "precision": 0.90}
        )
        result3 = ExperimentResult(
            name="exp3",
            pipeline=Pipeline([]),
            metrics={"accuracy": 0.90, "precision": 0.85}
        )

        evaluator.add_result(result1)
        evaluator.add_result(result2)
        evaluator.add_result(result3)

        best = evaluator.get_best()
        # result3 has highest accuracy (0.90), so it should be selected
        assert best == result3

    def test_get_best_multiple_metrics_min_mode(self):
        evaluator = Evaluator(metric=["mse", "mae"], mode="min")

        result1 = ExperimentResult(
            name="exp1",
            pipeline=Pipeline([]),
            metrics={"mse": 0.50, "mae": 0.40}
        )
        result2 = ExperimentResult(
            name="exp2",
            pipeline=Pipeline([]),
            metrics={"mse": 0.30, "mae": 0.35}
        )
        result3 = ExperimentResult(
            name="exp3",
            pipeline=Pipeline([]),
            metrics={"mse": 0.50, "mae": 0.30}
        )

        evaluator.add_result(result1)
        evaluator.add_result(result2)
        evaluator.add_result(result3)

        best = evaluator.get_best()
        # result2 has lowest mse (0.30), so it should be selected
        assert best == result2

    def test_add_multiple_results_and_get_best(self):
        import math
        evaluator = Evaluator(metric="f1_score", mode="max")

        results = []
        for i in range(5):
            result = ExperimentResult(
                name=f"exp{i}",
                pipeline=Pipeline([]),
                metrics={"f1_score": 0.70 + i * 0.05}
            )
            results.append(result)
            evaluator.add_result(result)

        best = evaluator.get_best()
        assert best == results[-1]
        assert best.metrics["f1_score"] != 0.9
