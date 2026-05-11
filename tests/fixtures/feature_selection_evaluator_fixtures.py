from unittest.mock import Mock, MagicMock

import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from app.core.context import Context, ProjectConfig
from app.core.enums import Stages
from app.core.stages.evaluation import FeatureSelectionEvaluator
from app.core.experiments import ExperimentResult


@pytest.fixture
def mock_config():
    config = Mock(spec=ProjectConfig)
    config.priority_metric = "test_accuracy"
    config.scoring = ["accuracy"]
    return config


@pytest.fixture
def mock_context(mock_config):
    context = Mock(spec=Context)
    context.config = mock_config
    context.stage_results = {}
    context.update_stage_context = MagicMock()
    return context


@pytest.fixture
def evaluator(mock_context):
    return FeatureSelectionEvaluator(
        stage=Stages.FEATURE_SELECTION, context=mock_context
    )

@pytest.fixture
def sample_results():
    results = []
    selectors = ["SelectKBest", "SelectKBest", "RFE", "RFE", "VarianceThreshold"]
    metrics = [0.95, 0.85, 0.93, 0.80, 0.90]

    for selector, metric in zip(selectors, metrics):
        pipeline = Pipeline([("feature_selection", SelectKBest(f_classif, k=3))])
        result = ExperimentResult(
            name=f"exp_{selector}_{metric}",
            pipeline=pipeline,
            metrics={"test_accuracy": metric},
            config={"selector": selector, "predictor": "LogisticRegression"},
            metadata={
                "selectors": [selector],
                "model": "LogisticRegression",
                "model_type": "LINEAR",
                "model_based": "TREE",
            },
            selected_features=["feature1", "feature2", "feature3"],
        )
        results.append(result)
    return results
