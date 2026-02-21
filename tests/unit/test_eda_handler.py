import pandas as pd

from app.core.context.init_context import init_context
from app.core.stages.data_inspection.data_inspection import DataInspectionStage


def test_detect_boolean_series():
    s = pd.Series([True, False, True])
    assert DataInspectionStage.detect_feature_type(s) == "binary"


def test_detect_binary_numeric():
    s = pd.Series([0, 1, 0, 1])
    assert DataInspectionStage.detect_feature_type(s) == "binary"


def test_detect_object_as_categorical():
    s = pd.Series(["a", "b", "c"])
    assert DataInspectionStage.detect_feature_type(s) == "categorical"


def test_numeric_high_unique_ratio_is_numerical():
    s = pd.Series(range(100))
    assert DataInspectionStage.detect_feature_type(s) == "numerical"


def test_detect_ordinal_semantics_true():
    s = pd.Series(["low", "medium", "high"])
    assert DataInspectionStage.detect_ordinal_semantics(s) is True


def test_detect_ordinal_semantics_false():
    s = pd.Series(["apple", "banana", "orange"])
    assert DataInspectionStage.detect_ordinal_semantics(s) is False


def test_ordinal_semantics_overrides_cardinality():
    s = pd.Series(["low", "medium", "high"])
    encoding = DataInspectionStage.decide_categorical_encoding(s, cardinality=3)
    assert encoding == "ordinal"


def test_low_cardinality_returns_onehot():
    s = pd.Series(["A", "B", "C"])
    encoding = DataInspectionStage.decide_categorical_encoding(s, cardinality=3)
    assert encoding == "onehot"


def test_high_cardinality_returns_ordinal():
    s = pd.Series([f"cat{i}" for i in range(20)])
    encoding = DataInspectionStage.decide_categorical_encoding(s, cardinality=20)
    assert encoding == "ordinal"


def test_no_transformation_when_low_skew():
    s = pd.Series([1, 2, 3, 4, 5])
    result = DataInspectionStage.analyze_numerical_distribution(s)
    assert result["suggested_transformation"] is None


def test_log_transformation_for_positive_skew():
    s = pd.Series([1] * 50 + [1000])
    result = DataInspectionStage.analyze_numerical_distribution(s)
    assert result["suggested_transformation"] == "log"


def test_yeo_johnson_for_negative_values():
    s = pd.Series([-10] * 50 + [1000])
    result = DataInspectionStage.analyze_numerical_distribution(s)
    assert result["suggested_transformation"] == "yeo-johnson"