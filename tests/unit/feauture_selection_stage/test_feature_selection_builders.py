import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet

from app.core.ml.pipeline_builder import PipelineBuilder
from app.core.stages.feature_selection.feature_selection import elasticnet_non_linear_model_builder, \
    FEATURE_SELECTION_EXPERIMENTS
from app.core.stages.feature_selection.feature_selection import elasticnet_selector_linear_model_builder
from app.core.stages.feature_selection.feature_selection import extra_trees_selector_linear_model_builder
from app.core.stages.feature_selection.feature_selection import extra_trees_selector_non_linear_model_builder


class TestFeatureSelectionBuilders:

    def setup_method(self):
        self.preprocessing = BaseEstimator()

    @pytest.mark.parametrize("builder", [
        elasticnet_selector_linear_model_builder,
        elasticnet_non_linear_model_builder,
        extra_trees_selector_linear_model_builder,
        extra_trees_selector_non_linear_model_builder,
    ])
    def test_steps_in_selectors_builders(self, builder):
        expected_steps = ["preprocessing", "feature_selection", "model"]

        builder = builder(self.preprocessing)
        steps = [step for step, _ in builder.steps]
        assert len(builder.steps) == len(expected_steps)
        assert steps == expected_steps

    def test_builders_return_pipeline_builder_instance(self):
        builder = extra_trees_selector_linear_model_builder(self.preprocessing)
        assert isinstance(builder, PipelineBuilder)

    @pytest.mark.parametrize("experiment", FEATURE_SELECTION_EXPERIMENTS)
    def test_feature_selector_step_matches_metadata(self, experiment):
        pipeline_builder = experiment.builder(self.preprocessing)

        # Convert steps to dict for easier access
        steps_dict = dict(pipeline_builder.steps)

        expected_selector = experiment.metadata["selector"]

        if expected_selector is None:
            # Should NOT contain feature_selection step
            assert "feature_selection" not in steps_dict
            return

        # Must contain feature_selection step
        assert "feature_selection" in steps_dict

        selector_step = steps_dict["feature_selection"]

        # Must be SelectFromModel
        assert isinstance(selector_step, SelectFromModel)

        model = selector_step.estimator

        if expected_selector == "ExtraTrees":
            assert isinstance(model, ExtraTreesClassifier)

        elif expected_selector == "ElasticNet":
            assert isinstance(model, ElasticNet)
        else:
            pytest.fail(f"Unknown selector type in metadata: {expected_selector}")
