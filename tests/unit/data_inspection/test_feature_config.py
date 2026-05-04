import pytest

from data_inspection import BooleanFeature


class TestBooleanFeature:

    def test_boolean_feature_init(self):
        assert BooleanFeature(name="test", dtype="bool") is not None

    def test_boolean_default_values(self):
        bool = BooleanFeature(name="test", dtype="bool")

        assert bool.true_ratio == 0.5
        assert bool.cast_to_int == True
        assert bool.is_imbalanced == False

    def test_boolean_custom_true_ratio(self):
        bool = BooleanFeature(name="test", dtype="bool", true_ratio=0.7)
        assert bool.true_ratio == 0.7
        assert bool.is_imbalanced == False

    @pytest.mark.parametrize("ratios, expected", [(0.1, False),
                                                  (0.04, True),
                                                  (0.99, True),
                                                  (0.05, False),
                                                  (0.95, False),
                                                  (0, True),
                                                  (1, True),
                                                  ],
                             ids=["low_ratio", "high_ratio", "equal_ratio", "low_ratio_2", "high_ratio_2", "zero_ratio",
                                  "one_ratio"])
    def test_boolean_is_imbalance(self, ratios: float | int, expected: bool):
        bool = BooleanFeature(name="test", dtype="bool", true_ratio=ratios)
        assert bool.is_imbalanced == expected

    def test_boolean_cast_to_int(self):
        bool = BooleanFeature(name="test", dtype="bool", cast_to_int=False)
        assert bool.cast_to_int == False

    def test_boolean_model_validate(self):
        model = {"name": "test", "dtype": "bool"}
        assert BooleanFeature.model_validate(model) == BooleanFeature(name="test", dtype="bool")
