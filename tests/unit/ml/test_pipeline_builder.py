import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.core.stages.data_inspection.pipeline_builder import PipelineBuilder


@pytest.mark.parametrize(
    "steps, expected_exception",
    [
        ([("preprocessing", "not an estimator")], TypeError),
        ([(1, BaseEstimator())], TypeError),
    ],
)
def test_pipeline_builder_invalid_steps(steps, expected_exception):
    with pytest.raises(expected_exception):
        PipelineBuilder(steps)


def test_pipeline_builder_empty_steps_allowed():
    builder = PipelineBuilder([])
    assert builder.steps == []


def test_pipeline_builder_none_steps_allowed():
    builder = PipelineBuilder(None)
    assert builder.steps == []


@pytest.mark.parametrize(
    "steps, expected_steps",
    [
        ([("preprocessing", BaseEstimator())], 1),
        ([("preprocessing", BaseEstimator()), ("classifier", BaseEstimator())], 2),
        ([("a", BaseEstimator()), ("b", BaseEstimator()), ("c", BaseEstimator())], 3),
    ],
)
def test_pipeline_builder_built_pipeline(steps, expected_steps):
    builder = PipelineBuilder(steps)
    assert len(builder.steps) == expected_steps
    builder.add_step(("extra", BaseEstimator()))
    assert len(builder.steps) == expected_steps + 1
    assert isinstance(builder.build(), Pipeline)


def test_pipeline_builder_add_step():
    builder = PipelineBuilder([("step1", BaseEstimator())])
    builder.add_step(("step2", BaseEstimator()))
    assert len(builder.steps) == 2


def test_pipeline_builder_add_step_invalid_name():
    builder = PipelineBuilder([("step1", BaseEstimator())])
    with pytest.raises(TypeError):
        builder.add_step((123, BaseEstimator()))


def test_pipeline_builder_add_step_invalid_estimator():
    builder = PipelineBuilder([("step1", BaseEstimator())])
    with pytest.raises(TypeError):
        builder.add_step(("step2", "not_an_estimator"))


def test_pipeline_builder_prepend_step():
    builder = PipelineBuilder([("step2", BaseEstimator())])
    builder.prepend_step(("step1", BaseEstimator()))
    assert builder.steps[0][0] == "step1"


def test_pipeline_builder_builds_sklearn_pipeline():
    steps = [("prep", BaseEstimator()), ("clf", BaseEstimator())]
    builder = PipelineBuilder(steps)
    pipeline = builder.build()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
