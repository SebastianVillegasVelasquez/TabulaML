from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from stages.data_inspection.feature_config import FeatureConfig
from stages.data_inspection.preprocessing_stage import PreprocessingBuilder


class TestPreprocessingBuilder:
    """Test suite for PreprocessingBuilder class"""

    def test_preprocessing_builder_creates_correct_transformers(self):
        """Test that the pipeline_builder creates the correct number of transformers"""
        feature_configs = [
            FeatureConfig(name="age", dtype="int64", feature_type="numerical"),
            FeatureConfig(
                name="gender", dtype="object", feature_type="categorical", encoding="onehot"
            ),
            FeatureConfig(
                name="grade", dtype="object", feature_type="categorical", encoding="ordinal"
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 3

    def test_numerical_features_only(self):
        """Test pipeline_builder with only numerical features"""
        feature_configs = [
            FeatureConfig(name="age", dtype="int64", feature_type="numerical", is_numerical=True),
            FeatureConfig(
                name="salary",
                dtype="float64",
                feature_type="numerical",
                is_numerical=True,
                skewness=1.5,
            ),
            FeatureConfig(
                name="experience", dtype="int64", feature_type="numerical", is_numerical=True
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 1
        assert transformer.transformers[0][0] == "num"
        assert transformer.transformers[0][2] == ["age", "salary", "experience"]

        # Check pipeline steps
        pipeline = transformer.transformers[0][1]
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.steps[0][1], SimpleImputer)
        assert isinstance(pipeline.steps[1][1], StandardScaler)

    def test_categorical_onehot_features_only(self):
        """Test pipeline_builder with only one-hot encoded categorical features"""
        feature_configs = [
            FeatureConfig(
                name="gender",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
                cardinality=2,
            ),
            FeatureConfig(
                name="city",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
                cardinality=10,
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 1
        assert transformer.transformers[0][0] == "cat_nominal"
        assert transformer.transformers[0][2] == ["gender", "city"]

        # Check pipeline steps
        pipeline = transformer.transformers[0][1]
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.steps[0][1], SimpleImputer)
        assert pipeline.steps[0][1].strategy == "most_frequent"
        assert isinstance(pipeline.steps[1][1], OneHotEncoder)
        assert pipeline.steps[1][1].handle_unknown == "ignore"

    def test_categorical_ordinal_features_only(self):
        """Test pipeline_builder with only ordinal encoded categorical features"""
        feature_configs = [
            FeatureConfig(
                name="education",
                dtype="object",
                feature_type="categorical",
                encoding="ordinal",
                is_categorical=True,
                cardinality=5,
            ),
            FeatureConfig(
                name="grade",
                dtype="object",
                feature_type="categorical",
                encoding="ordinal",
                is_categorical=True,
                cardinality=3,
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 1
        assert transformer.transformers[0][0] == "cat_ordinal"
        assert transformer.transformers[0][2] == ["education", "grade"]

        # Check pipeline steps
        pipeline = transformer.transformers[0][1]
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.steps[0][1], SimpleImputer)
        assert pipeline.steps[0][1].strategy == "most_frequent"
        assert isinstance(pipeline.steps[1][1], OrdinalEncoder)
        assert pipeline.steps[1][1].handle_unknown == "use_encoded_value"
        assert pipeline.steps[1][1].unknown_value == -1

    def test_mixed_feature_types(self):
        """Test pipeline_builder with mixed feature types"""
        feature_configs = [
            FeatureConfig(name="age", dtype="int64", feature_type="numerical", is_numerical=True),
            FeatureConfig(
                name="gender",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
            ),
            FeatureConfig(
                name="education",
                dtype="object",
                feature_type="categorical",
                encoding="ordinal",
                is_categorical=True,
            ),
            FeatureConfig(
                name="salary",
                dtype="float64",
                feature_type="numerical",
                is_numerical=True,
                skewness=2.1,
            ),
            FeatureConfig(
                name="city",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
                cardinality=50,
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 3

        # Check numerical transformer
        num_transformer = next(t for t in transformer.transformers if t[0] == "num")
        assert num_transformer[2] == ["age", "salary"]

        # Check onehot transformer
        onehot_transformer = next(t for t in transformer.transformers if t[0] == "cat_nominal")
        assert onehot_transformer[2] == ["gender", "city"]

        # Check ordinal transformer
        ordinal_transformer = next(t for t in transformer.transformers if t[0] == "cat_ordinal")
        assert ordinal_transformer[2] == ["education"]

    def test_empty_feature_configs(self):
        """Test pipeline_builder with empty feature configs"""
        feature_configs = []

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 0

    def test_feature_with_suggested_transformation(self):
        """Test pipeline_builder handles features with suggested transformations"""
        feature_configs = [
            FeatureConfig(
                name="income",
                dtype="float64",
                feature_type="numerical",
                is_numerical=True,
                skewness=3.5,
                suggested_transformation="log",
            ),
            FeatureConfig(
                name="age", dtype="int64", feature_type="numerical", is_numerical=True, skewness=0.2
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 1
        assert transformer.transformers[0][2] == ["income", "age"]

    def test_feature_with_high_cardinality(self):
        """Test pipeline_builder handles high cardinality categorical features"""
        feature_configs = [
            FeatureConfig(
                name="user_id",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
                cardinality=10000,
            ),
            FeatureConfig(
                name="category",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
                cardinality=5,
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 1
        assert transformer.transformers[0][2] == ["user_id", "category"]

    def test_feature_with_zero_ratio(self):
        """Test pipeline_builder handles features with high zero ratio"""
        feature_configs = [
            FeatureConfig(
                name="purchases",
                dtype="int64",
                feature_type="numerical",
                is_numerical=True,
                zero_ratio=0.95,
            ),
            FeatureConfig(
                name="clicks",
                dtype="int64",
                feature_type="numerical",
                is_numerical=True,
                zero_ratio=0.1,
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert len(transformer.transformers) == 1
        assert transformer.transformers[0][2] == ["purchases", "clicks"]

    def test_categorical_without_encoding_specified(self):
        """Test pipeline_builder handles categorical features without encoding specification"""
        feature_configs = [
            FeatureConfig(
                name="color",
                dtype="object",
                feature_type="categorical",
                is_categorical=True,
                cardinality=3,
            ),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        # Should not create any transformer since encoding is None
        assert len(transformer.transformers) == 0

    def test_column_transformer_returns_correct_type(self):
        """Test that build returns ColumnTransformer instance"""
        feature_configs = [
            FeatureConfig(name="age", dtype="int64", feature_type="numerical"),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer = builder.build()

        assert isinstance(transformer, ColumnTransformer)

    def test_builder_does_not_modify_feature_configs(self):
        """Test that pipeline_builder does not modify input feature configs"""
        feature_configs = [
            FeatureConfig(name="age", dtype="int64", feature_type="numerical", is_numerical=True),
            FeatureConfig(
                name="gender",
                dtype="object",
                feature_type="categorical",
                encoding="onehot",
                is_categorical=True,
            ),
        ]

        original_configs = [(fc.name, fc.feature_type, fc.encoding) for fc in feature_configs]

        builder = PreprocessingBuilder(feature_configs)
        builder.build()

        # Verify configs unchanged
        for i, fc in enumerate(feature_configs):
            assert (fc.name, fc.feature_type, fc.encoding) == original_configs[i]

    def test_multiple_builds_return_independent_transformers(self):
        """Test that calling build multiple times returns independent transformers"""
        feature_configs = [
            FeatureConfig(name="age", dtype="int64", feature_type="numerical"),
        ]

        builder = PreprocessingBuilder(feature_configs)
        transformer1 = builder.build()
        transformer2 = builder.build()

        assert transformer1 is not transformer2
        assert len(transformer1.transformers) == len(transformer2.transformers)
