from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.utils.logger import logger
from .feature_config import FeatureConfig


class PreprocessingBuilder:
    """
    Builds a sklearn ColumnTransformer based on FeatureConfig metadata.
    This class should DO NOT fit or transform data.
    """

    def __init__(self, feature_configs: list[FeatureConfig]):
        self.feature_configs = feature_configs
        self.num_features: list[FeatureConfig] = []
        self.cat_nominal: list[FeatureConfig] = []
        self.cat_ordinal: list[FeatureConfig] = []
        self.transformers: list[tuple[str, Pipeline]] = []

    def build(self) -> ColumnTransformer:
        """
        Builds and returns a ColumnTransformer based on feature metadata.

        :return: ColumnTransformer instance.
        """

        self._steps_orchestator()

        return ColumnTransformer(transformers=self.transformers)

    def _steps_orchestator(self):
        # Group features by type
        self._group_features_by_type()

        # Identify same steps for each feature type group
        self._identify_and_group_same_steps()

    def _identify_and_group_same_steps(self):
        # Handle the numerical logic
        self._handle_numerical_features()

    def _handle_numerical_features(self
                                   ) -> list[tuple[str, Pipeline]]:

        assert self.num_features, "No numerical features found"

        unique_steps = set()

        for feature in self.num_features:
            logger.info(f"feature: {feature.name},"
                        f"feature suggested scaler {feature.suggested_scaler},"
                        f"imputer {feature.suggested_imputer},"
                        f"suggested transformer {feature.suggested_transformation} "
                        f"which encoding needs: {feature.which_encoding_needs}")

    def _group_features_by_type(self) -> None:
        """ Group features by their type (numerical, categorical, etc.)

        Once the data inspection stage is complete, the feature_configs list is populated.
        This method groups the features by their type and stores them in separate lists.
        """
        from .feature_config_enum import FeatureType

        for feature in self.feature_configs:
            match feature.feature_type:
                case FeatureType.NUMERICAL:
                    self.num_features.append(feature)
                case FeatureType.CATEGORICAL_NOMINAL:
                    self.cat_nominal.append(feature)
                case FeatureType.CATEGORICAL_ORDINAL:
                    self.cat_ordinal.append(feature)
