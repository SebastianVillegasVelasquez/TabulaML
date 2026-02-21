from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.ml.pipeline_builder import PipelineBuilder


def logistic_builder(preprocessing):
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )


def random_forest_builder(preprocessing):
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("model", RandomForestClassifier(n_estimators=200))
        ]
    )


MODEL_SELECTION_EXPERIMENTS = [

    ExperimentDefinition(
        name="logistic_regression",
        stage="model_selection",
        builder=logistic_builder,
        metadata={"model": "LogisticRegression"}
    ),

    ExperimentDefinition(
        name="random_forest",
        stage="model_selection",
        builder=random_forest_builder,
        metadata={"model": "RandomForestClassifier"}
    ),

]
