from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, LogisticRegression

from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.ml.pipeline_builder import PipelineBuilder

"""
The main idea behind this stage is to evaluate the impact of different 
feature selection methods on the model performance.

We will compare the following approaches:
1. No feature selection (baseline)
2. Feature selection using ExtraTreesClassifier (tree-based feature importance)
3. Feature selection using ElasticNet (linear model with L1 regularization)

For each feature selection method, we will evaluate both linear and non-linear models 
to see how the choice of predictor interacts with the selected features.
"""


# Base model builder for all linear experiments in this stage
def build_base_linear_model():
    return LogisticRegression(
        max_iter=5000,
        C=1.0,
        solver="lbfgs"
    )


# Base model builder for all non-linear experiments in this stage
def build_base_non_linear_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


# Experiments using linear models for feature selection
def extra_trees_selector_linear_model_builder(preprocessing):
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
            ("model", build_base_linear_model())
        ]
    )


def elasticnet_selector_linear_model_builder(preprocessing):
    selector = SelectFromModel(
        ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42
        ),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


# Experiments using non-linear models for feature selection
def extra_trees_selector_non_linear_model_builder(preprocessing):
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
            ("model", build_base_non_linear_model())
        ]
    )


def elasticnet_non_linear_model_builder(preprocessing):
    selector = SelectFromModel(
        ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42
        ),
        threshold="median"
    )
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


# Experiment performing no feature selection, only preprocessing and modeling
def no_selector_builder(preprocessing):
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("model", build_base_linear_model())
        ]
    )


FEATURE_SELECTION_EXPERIMENTS = [

    ExperimentDefinition(
        name="no_selector",
        stage="feature_selection",
        builder=no_selector_builder,
        metadata={"selector": None, "predictor": None}
    ),
    ExperimentDefinition(
        name="extra_trees_selector_linear_model",
        stage="feature_selection",
        builder=extra_trees_selector_linear_model_builder,
        metadata={"selector": "ExtraTrees", "predictor": "Linear"}
    ),
    ExperimentDefinition(
        name="elasticnet_selector_linear_model",
        stage="feature_selection",
        builder=elasticnet_selector_linear_model_builder,
        metadata={"selector": "ElasticNet", "predictor": "Linear"}
    ),
    ExperimentDefinition(
        name="extra_trees_selector_non_linear_model",
        stage="feature_selection",
        builder=extra_trees_selector_non_linear_model_builder,
        metadata={"selector": "ExtraTrees", "predictor": "NonLinear"}
    ),
    ExperimentDefinition(
        name="elastic_net_non_linear_model",
        stage="feature_selection",
        builder=elasticnet_non_linear_model_builder,
        metadata={"selector": "ElasticNet", "predictor": "NonLinear"}
    ),
]
