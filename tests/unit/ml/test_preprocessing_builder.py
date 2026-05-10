"""Tests for PreprocessingBuilder using the current FeatureConfig subclass API."""

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)

from app.core.stages.data_inspection import (
    NumericalFeature,
    CategoricalNominalFeature,
    CategoricalOrdinalFeature,
)
from app.core.stages.data_inspection.preprocessing_stage import PreprocessingBuilder


def make_numerical(name, skewness=0.0, **kwargs):
    return NumericalFeature(name=name, dtype="float64", skewness=skewness, **kwargs)


def make_nominal(name, cardinality=5, **kwargs):
    return CategoricalNominalFeature(
        name=name, dtype="object", cardinality=cardinality, **kwargs
    )


def make_ordinal(name, cardinality=3, **kwargs):
    return CategoricalOrdinalFeature(
        name=name, dtype="object", cardinality=cardinality, **kwargs
    )


class TestPreprocessingBuilderNumerical:
    def test_single_numerical_feature(self):
        features = [make_numerical("age", skewness=0.0)]
        transformer = PreprocessingBuilder(features).build()
        assert isinstance(transformer, ColumnTransformer)
        names = [t[0] for t in transformer.transformers]
        assert any("num" in n for n in names)

    def test_multiple_numerical_features(self):
        features = [
            make_numerical("age", skewness=0.0),
            make_numerical("income", skewness=0.0),
        ]
        transformer = PreprocessingBuilder(features).build()
        all_cols = [col for t in transformer.transformers for col in t[2]]
        assert "age" in all_cols
        assert "income" in all_cols

    def test_skewed_numerical_uses_robust_scaler(self):
        features = [make_numerical("income", skewness=2.5)]
        transformer = PreprocessingBuilder(features).build()
        pipeline = transformer.transformers[0][1]
        scaler = pipeline.named_steps.get("scaler")
        assert isinstance(scaler, RobustScaler)

    def test_clean_numerical_uses_standard_scaler(self):
        features = [make_numerical("age", skewness=0.0)]
        transformer = PreprocessingBuilder(features).build()
        pipeline = transformer.transformers[0][1]
        scaler = pipeline.named_steps.get("scaler")
        assert isinstance(scaler, StandardScaler)

    def test_numerical_pipeline_has_imputer(self):
        features = [make_numerical("age", skewness=0.0)]
        transformer = PreprocessingBuilder(features).build()
        pipeline = transformer.transformers[0][1]
        assert "imputer" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)

    def test_returns_column_transformer(self):
        features = [make_numerical("x", skewness=0.0)]
        result = PreprocessingBuilder(features).build()
        assert isinstance(result, ColumnTransformer)

    def test_multiple_builds_return_independent_transformers(self):
        features = [make_numerical("age", skewness=0.0)]
        builder = PreprocessingBuilder(features)
        t1 = builder.build()
        t2 = builder.build()
        assert t1 is not t2
        assert len(t1.transformers) == len(t2.transformers)


class TestPreprocessingBuilderCategorical:
    def test_nominal_feature_uses_onehot(self):
        features = [make_nominal("color", cardinality=3)]
        transformer = PreprocessingBuilder(features).build()
        names = [t[0] for t in transformer.transformers]
        assert any("cat_nominal" in n for n in names)

    def test_nominal_pipeline_has_onehot_encoder(self):
        features = [make_nominal("city", cardinality=5)]
        transformer = PreprocessingBuilder(features).build()
        pipeline = transformer.transformers[0][1]
        steps = dict(pipeline.steps)
        assert "encoder" in steps
        assert isinstance(steps["encoder"], OneHotEncoder)

    def test_ordinal_feature_has_ordinal_encoder(self):
        features = [make_ordinal("grade", cardinality=3)]
        transformer = PreprocessingBuilder(features).build()
        names = [t[0] for t in transformer.transformers]
        assert any("cat_ordinal" in n for n in names)

    def test_ordinal_pipeline_has_ordinal_encoder(self):
        features = [make_ordinal("education", cardinality=4)]
        transformer = PreprocessingBuilder(features).build()
        pipeline = transformer.transformers[0][1]
        steps = dict(pipeline.steps)
        assert "encoder" in steps
        assert isinstance(steps["encoder"], OrdinalEncoder)

    def test_nominal_imputer_uses_most_frequent(self):
        features = [make_nominal("category", cardinality=5)]
        transformer = PreprocessingBuilder(features).build()
        pipeline = transformer.transformers[0][1]
        imputer = pipeline.named_steps["imputer"]
        assert imputer.strategy == "most_frequent"

    def test_multiple_nominal_features(self):
        features = [
            make_nominal("color", cardinality=3),
            make_nominal("size", cardinality=4),
        ]
        transformer = PreprocessingBuilder(features).build()
        all_cols = [col for t in transformer.transformers for col in t[2]]
        assert "color" in all_cols
        assert "size" in all_cols


class TestPreprocessingBuilderMixed:
    def test_numerical_and_nominal(self):
        features = [
            make_numerical("age", skewness=0.0),
            make_nominal("city", cardinality=5),
        ]
        transformer = PreprocessingBuilder(features).build()
        assert len(transformer.transformers) == 2
        names = [t[0] for t in transformer.transformers]
        assert any("num" in n for n in names)
        assert any("cat" in n for n in names)

    def test_numerical_nominal_ordinal(self):
        features = [
            make_numerical("salary", skewness=0.0),
            make_nominal("country", cardinality=3),
            make_ordinal("education", cardinality=4),
        ]
        transformer = PreprocessingBuilder(features).build()
        assert len(transformer.transformers) == 3

    def test_builder_does_not_modify_input_names(self):
        features = [
            make_numerical("age", skewness=0.0),
            make_nominal("gender", cardinality=2),
        ]
        original_names = [f.name for f in features]
        PreprocessingBuilder(features).build()
        assert [f.name for f in features] == original_names


class TestPreprocessingBuilderEdgeCases:
    def test_empty_feature_configs_raises(self):
        builder = PreprocessingBuilder([])
        with pytest.raises(ValueError):
            builder.build()

    def test_identifier_feature_is_skipped(self):
        from app.core.stages.data_inspection import IdentifierFeature

        features = [
            IdentifierFeature(name="id", dtype="int64", cardinality=100),
            make_numerical("age", skewness=0.0),
        ]
        transformer = PreprocessingBuilder(features).build()
        all_cols = [col for t in transformer.transformers for col in t[2]]
        assert "id" not in all_cols
        assert "age" in all_cols

    def test_drop_feature_is_skipped(self):
        features = [
            make_numerical("age", skewness=0.0),
            make_numerical("dropped", skewness=0.0, drop=True),
        ]
        transformer = PreprocessingBuilder(features).build()
        all_cols = [col for t in transformer.transformers for col in t[2]]
        assert "dropped" not in all_cols
        assert "age" in all_cols
