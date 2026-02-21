import pytest

from sklearn.base import BaseEstimator


class TestFeatureSelectionBuilders:

    def setup_method(self):
        self.preprocessing = BaseEstimator()

    @pytest.mark.parametrize("expected_steps", [["preprocessing", "feature_selection", "model"]])
    def test_selector_builders(self, expected_steps):
        from app.core.domain.experiments.feature_selection import elasticnet_selector_builder
        builder = elasticnet_selector_builder(self.preprocessing)
        steps = [step for step, _ in builder.steps]
        assert len(builder.steps) == len(expected_steps)
        assert steps == expected_steps
#
