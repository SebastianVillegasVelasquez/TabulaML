import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_config import FeatureConfig


class PreprocessingBuilder:
    """
    Builds a sklearn ColumnTransformer based on FeatureConfig metadata.
    This class should DO NOT fit or transform data.
    """

    def __init__(self, feature_configs: list[FeatureConfig]):
        self.feature_configs = feature_configs
        self.num_features: list[FeatureConfig] = []
        self.cat_onehot: list[FeatureConfig] = []
        self.cat_ordinal: list[FeatureConfig] = []
        self.transformers: list[tuple[str, Pipeline]] = []

    def build(self) -> ColumnTransformer:
        """
        Builds and returns a ColumnTransformer based on feature metadata.

        :return: ColumnTransformer instance.
        """
        try:
            self._identify_and_load_features()
        except ValueError as e:
            raise ValueError(f"Error identifying features: {e}") from e


        self._add_transformers()



        return ColumnTransformer(transformers=transformers)

    def _add_transformers(self):
        pass



    @staticmethod
    def _scale_numerical_features(X: pd.DataFrame,
                                  scaler: StandardScaler,
                                  ) -> pd.DataFrame:
        scaled = scaler.fit_transform(X)
        return pd.DataFrame(scaled, columns=X.columns)

    def _identify_and_load_features(self):
        if self.feature_configs is None:
            raise ValueError("Feature configs are not provided.")

        self.num_features = []
        self.cat_onehot = []
        self.cat_ordinal = []

        for fc in self.feature_configs:
            match fc.feature_type:
                case "numerical":
                    self.num_features.append(fc)

                case "categorical":
                    if fc.encoding == "onehot":
                        self.cat_onehot.append(fc)
                    elif fc.encoding == "ordinal":
                        self.cat_ordinal.append(fc)
                    else:
                        raise ValueError(
                            f"Unsupported encoding: {fc.encoding}"
                        )

                case _:
                    raise ValueError(
                        f"Unsupported feature type: {fc.feature_type}"
                    )