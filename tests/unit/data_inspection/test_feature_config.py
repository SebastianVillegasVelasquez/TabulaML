import pytest
from pydantic import ValidationError

from app.core.stages.data_inspection import BooleanFeature, FeatureConfig, FeatureType


class TestFeatureConfig:

    def test_init_success(self, feature_config):
        assert feature_config.name == "age"
        assert feature_config.dtype == "int64"
        assert feature_config.feature_type == FeatureType.NUMERICAL
        assert feature_config.missing_ratio == 0.2
        assert feature_config.is_target is False
        assert feature_config.drop is False
        assert feature_config.notes == "Test feature"

    def test_defaults(self):
        feature = FeatureConfig(name="test", dtype="float64", feature_type=FeatureType.NUMERICAL)

        assert feature.missing_ratio == 0.0
        assert feature.is_target is False
        assert feature.drop is False
        assert feature.notes is None

    def test_needs_imputation_true(self, feature_config):
        assert feature_config.needs_imputation is True

    def test_needs_imputation_false(self):
        feature = FeatureConfig(
            name="test", dtype="float64", feature_type=FeatureType.NUMERICAL, missing_ratio=0.0
        )
        assert feature.needs_imputation is False

    def test_is_high_missing_true(self):
        feature = FeatureConfig(
            name="test", dtype="float64", feature_type=FeatureType.NUMERICAL, missing_ratio=0.7
        )
        assert feature.is_high_missing is True

    def test_is_high_missing_false(self):
        feature = FeatureConfig(
            name="test", dtype="float64", feature_type=FeatureType.NUMERICAL, missing_ratio=0.5
        )
        assert feature.is_high_missing is False

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_missing_ratio_edge_values(self, value):
        feature = FeatureConfig(
            name="test", dtype="float64", feature_type=FeatureType.NUMERICAL, missing_ratio=value
        )
        assert feature.missing_ratio == value

    def test_is_high_missing_boundary(self):
        feature = FeatureConfig(
            name="test", dtype="float64", feature_type=FeatureType.NUMERICAL, missing_ratio=0.6
        )
        assert feature.is_high_missing is False

    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
    def test_invalid_missing_ratio(self, invalid_value):
        with pytest.raises(ValidationError):
            FeatureConfig(
                name="test",
                dtype="float64",
                feature_type=FeatureType.NUMERICAL,
                missing_ratio=invalid_value,
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            FeatureConfig(
                name="test",
                dtype="float64",
                feature_type=FeatureType.NUMERICAL,
                unknown_field="boom",
            )

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            FeatureConfig(name=123, dtype="float64", feature_type=FeatureType.NUMERICAL)

    def test_validate_assignment_success(self, feature_config):
        feature_config.missing_ratio = 0.5
        assert feature_config.missing_ratio == 0.5

    def test_validate_assignment_failure(self, feature_config):
        with pytest.raises(ValidationError):
            feature_config.missing_ratio = 2.0

    def test_drop_flag(self, feature_config):
        feature_config.drop = True
        assert feature_config.drop is True

    def test_target_flag(self):
        feature = FeatureConfig(
            name="target", dtype="int64", feature_type=FeatureType.NUMERICAL, is_target=True
        )
        assert feature.is_target is True


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

    @pytest.mark.parametrize(
        "ratios, expected",
        [
            (0.1, False),
            (0.04, True),
            (0.99, True),
            (0.05, False),
            (0.95, False),
            (0, True),
            (1, True),
        ],
        ids=[
            "low_ratio",
            "high_ratio",
            "equal_ratio",
            "low_ratio_2",
            "high_ratio_2",
            "zero_ratio",
            "one_ratio",
        ],
    )
    def test_boolean_is_imbalance(self, ratios: float | int, expected: bool):
        bool = BooleanFeature(name="test", dtype="bool", true_ratio=ratios)
        assert bool.is_imbalanced == expected

    def test_boolean_cast_to_int(self):
        bool = BooleanFeature(name="test", dtype="bool", cast_to_int=False)
        assert bool.cast_to_int == False

    def test_boolean_model_validate(self):
        model = {"name": "test", "dtype": "bool"}
        assert BooleanFeature.model_validate(model) == BooleanFeature(name="test", dtype="bool")
