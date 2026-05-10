import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from app.core.experiments import ExperimentResult


@pytest.fixture
def simple_pipeline():
    return Pipeline([("classifier", LogisticRegression())])


@pytest.fixture
def rf_pipeline():
    return Pipeline([("classifier", RandomForestClassifier(n_estimators=5))])


@pytest.fixture
def experiment_result_lr(simple_pipeline):
    return ExperimentResult(
        name="lr_exp",
        pipeline=simple_pipeline,
        metrics={"test_f1": 0.85, "test_accuracy": 0.87},
        config={"model": "LogisticRegression"},
        metadata={},
        selected_features=["f1", "f2"],
    )


@pytest.fixture
def experiment_result_rf(rf_pipeline):
    return ExperimentResult(
        name="rf_exp",
        pipeline=rf_pipeline,
        metrics={"test_f1": 0.92, "test_accuracy": 0.93},
        config={"model": "RandomForestClassifier"},
        metadata={},
        selected_features=["f1", "f2"],
    )


@pytest.fixture
def sample_experiment_results():
    results = []
    for i, (model, acc) in enumerate(
        [
            ("LogisticRegression", 0.85),
            ("RandomForest", 0.95),
            ("SVM", 0.90),
        ]
    ):
        results.append(
            ExperimentResult(
                name=f"exp_{i}",
                pipeline=Pipeline([("clf", LogisticRegression())]),
                metrics={"test_f1": acc, "test_accuracy": acc},
                config={"model": model},
                metadata={},
            )
        )
    return results
