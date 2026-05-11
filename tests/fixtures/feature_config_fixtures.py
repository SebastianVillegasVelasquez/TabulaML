import pytest

from app.core.stages.data_inspection import FeatureConfig, FeatureType


@pytest.fixture
def valid_feature_config_data():
    return {
        "name": "age",
        "dtype": "int64",
        "feature_type": FeatureType.NUMERICAL,
        "missing_ratio": 0.2,
        "is_target": False,
        "drop": False,
        "notes": "Test feature",
    }


@pytest.fixture
def feature_config(valid_feature_config_data):
    return FeatureConfig(**valid_feature_config_data)
