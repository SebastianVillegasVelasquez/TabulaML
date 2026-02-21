import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.core.ml.pipeline_builder import PipelineBuilder


@pytest.mark.parametrize("steps, expected_exception", [

    ([], ValueError),
    ([("preprocessing", "not an estimator")], TypeError),
    ([(1, BaseEstimator())], TypeError),

])
def test_pipeline_builder(steps, expected_exception):
    with pytest.raises(expected_exception):
        PipelineBuilder(steps)


@pytest.mark.parametrize("steps, expected_steps", [
    ([("preprocessing", BaseEstimator())], 1),
    ([("preprocessing", BaseEstimator()), ("classifier", BaseEstimator())], 2),
    ([("preprocessing", BaseEstimator()), ("classifier", BaseEstimator()), ("clustering", BaseEstimator())], 3),
])
def test_pipeline_builder_built_pipeline(steps, expected_steps):
    builder = PipelineBuilder(steps)
    assert len(builder.steps) == expected_steps
    builder.add_step(("preprocessing", BaseEstimator()))
    assert len(builder.steps) == expected_steps + 1
    assert isinstance(builder.build(), Pipeline)


