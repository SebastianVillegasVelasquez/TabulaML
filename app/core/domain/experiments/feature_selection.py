from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, LogisticRegression

from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.ml.pipeline_builder import PipelineBuilder

model = LogisticRegression(max_iter=1000, C=1.0)


def elasticnet_selector_builder(preprocessing):
    selector = SelectFromModel(
        ElasticNet(alpha=0.1)
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", model)
        ]
    )


def l1_logistic_selector_builder(preprocessing):
    selector = SelectFromModel(
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000
        )
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", model)
        ]
    )


def random_forest_selector_builder(preprocessing):
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", model)
        ]
    )


def extratrees_selector_builder(preprocessing):
    selector = SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=200,
            random_state=42
        ),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", model)
        ]
    )


def no_selector_builder(preprocessing):
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model)
        ]
    )


FEATURE_SELECTION_EXPERIMENTS = [

    ExperimentDefinition(
        name="no_selector",
        stage="feature_selection",
        builder=no_selector_builder,
        metadata={"selector": None}
    ),

    ExperimentDefinition(
        name="elasticnet_selector",
        stage="feature_selection",
        builder=elasticnet_selector_builder,
        metadata={"selector": "ElasticNet", "alpha": 0.1}
    ),

    ExperimentDefinition(
        name="l1_logistic_selector",
        stage="feature_selection",
        builder=l1_logistic_selector_builder,
        metadata={"selector": "LogisticL1"}
    ),

    ExperimentDefinition(
        name="rf_selector",
        stage="feature_selection",
        builder=random_forest_selector_builder,
        metadata={"selector": "RandomForest"}
    ),

    ExperimentDefinition(
        name="extratrees_selector",
        stage="feature_selection",
        builder=extratrees_selector_builder,
        metadata={"selector": "ExtraTrees"}
    ),
]
