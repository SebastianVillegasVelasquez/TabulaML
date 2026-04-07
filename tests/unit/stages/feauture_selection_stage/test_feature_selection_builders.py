import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from app.core.ml.pipeline_builder import PipelineBuilder
from app.core.stages.feature_selection.feature_selection_experiments import (
    FEATURE_SELECTION_EXPERIMENTS,
    no_selector_linear_builder,
    no_selector_nonlinear_builder,
    selectkbest_f_classif_linear_builder,
    selectkbest_mutual_info_nonlinear_builder,
    lasso_selector_linear_builder,
    elasticnet_selector_linear_builder,
    elasticnet_selector_nonlinear_builder,
    extratrees_selector_linear_builder,
    extratrees_selector_nonlinear_builder,
    rfe_linear_builder,
    rfe_nonlinear_builder,
)


class TestFeatureSelectionBuilders:

    def setup_method(self):
        self.preprocessing = BaseEstimator()

    @pytest.mark.parametrize("pipeline_builder,expected_steps", [
        (no_selector_linear_builder, ["preprocessing", "model"]),
        (no_selector_nonlinear_builder, ["preprocessing", "model"]),
        (selectkbest_f_classif_linear_builder, ["preprocessing", "feature_selection", "model"]),
        (selectkbest_mutual_info_nonlinear_builder, ["preprocessing", "feature_selection", "model"]),
        (lasso_selector_linear_builder, ["preprocessing", "feature_selection", "model"]),
        (elasticnet_selector_linear_builder, ["preprocessing", "feature_selection", "model"]),
        (elasticnet_selector_nonlinear_builder, ["preprocessing", "feature_selection", "model"]),
        (extratrees_selector_linear_builder, ["preprocessing", "feature_selection", "model"]),
        (extratrees_selector_nonlinear_builder, ["preprocessing", "feature_selection", "model"]),
        (rfe_linear_builder, ["preprocessing", "feature_selection", "model"]),
        (rfe_nonlinear_builder, ["preprocessing", "feature_selection", "model"]),
    ])
    def test_steps_in_selectors_builders(self, builder, expected_steps):
        pipeline_builder = builder(self.preprocessing)
        steps = [step for step, _ in pipeline_builder.steps]
        assert len(pipeline_builder.steps) == len(expected_steps)
        assert steps == expected_steps

    def test_builders_return_pipeline_builder_instance(self):
        builder = extratrees_selector_linear_builder(self.preprocessing)
        assert isinstance(builder, PipelineBuilder)

    @pytest.mark.parametrize("experiment", FEATURE_SELECTION_EXPERIMENTS)
    def test_all_experiments_return_pipeline_builder(self, experiment):
        pipeline_builder = experiment.pipeline_builder(self.preprocessing)
        assert isinstance(pipeline_builder, PipelineBuilder)

    @pytest.mark.parametrize("experiment", FEATURE_SELECTION_EXPERIMENTS)
    def test_feature_selector_step_matches_metadata(self, experiment):
        pipeline_builder = experiment.pipeline_builder(self.preprocessing)

        # Convert steps to dict for easier access
        steps_dict = dict(pipeline_builder.steps)

        expected_selector = experiment.metadata["selector"]

        if expected_selector == "none":
            # Should NOT contain feature_selection step for baseline
            assert "feature_selection" not in steps_dict
            return

        # All other experiments must contain feature_selection step
        assert "feature_selection" in steps_dict

        selector_step = steps_dict["feature_selection"]

        # Validate selector type based on metadata
        if expected_selector.startswith("selectkbest"):
            assert isinstance(selector_step, SelectKBest)
        elif expected_selector in ["lasso", "elasticnet", "extratrees"]:
            assert isinstance(selector_step, SelectFromModel)
            # Validate the wrapped estimator
            if expected_selector == "lasso":
                assert isinstance(selector_step.estimator, Lasso)
            elif expected_selector == "elasticnet":
                assert isinstance(selector_step.estimator, ElasticNet)
            elif expected_selector == "extratrees":
                assert isinstance(selector_step.estimator, ExtraTreesClassifier)
        elif expected_selector.startswith("rfe"):
            assert isinstance(selector_step, RFE)
            # Validate the wrapped estimator
            if expected_selector == "rfe_linear":
                assert isinstance(selector_step.estimator, LogisticRegression)
            elif expected_selector == "rfe_tree":
                assert isinstance(selector_step.estimator, DecisionTreeClassifier)
        else:
            pytest.fail(f"Unknown selector type in metadata: {expected_selector}")

    @pytest.mark.parametrize("experiment", FEATURE_SELECTION_EXPERIMENTS)
    def test_all_experiments_have_required_metadata(self, experiment):
        """Test that all experiments have the required metadata fields"""
        assert "selector" in experiment.metadata
        assert "selector_type" in experiment.metadata
        assert "validator" in experiment.metadata
        assert experiment.metadata["selector_type"] in ["baseline", "statistical", "l1_based", "tree_based", "wrapper"]
        assert experiment.metadata["validator"] in ["linear", "nonlinear"]

    @pytest.mark.parametrize("experiment", FEATURE_SELECTION_EXPERIMENTS)
    def test_all_experiments_have_model_step(self, experiment):
        """Test that all experiments have a model step"""
        pipeline_builder = experiment.pipeline_builder(self.preprocessing)
        steps_dict = dict(pipeline_builder.steps)
        assert "model" in steps_dict
        assert steps_dict["model"] is not None
