from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from app.core.domain.feature_config import FeatureConfig


class PreprocessingBuilder:
    """
    Builds a sklearn ColumnTransformer based on FeatureConfig metadata.
    This class should DO NOT fit or transform data.
    """

    def __init__(self, feature_configs: list[FeatureConfig]):
        self.feature_configs = feature_configs

    def build(self) -> ColumnTransformer:
        """
        Builds and returns a ColumnTransformer based on feature metadata.

        :return: ColumnTransformer instance.
        """
        num_features = []
        cat_onehot = []
        cat_ordinal = []

        for fc in self.feature_configs:
            if fc.feature_type == "numerical":
                num_features.append(fc.name)

            elif fc.feature_type == "categorical":
                if fc.encoding == "onehot":
                    cat_onehot.append(fc.name)
                elif fc.encoding == "ordinal":
                    cat_ordinal.append(fc.name)

        transformers = []

        if num_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_features,
                )
            )

        if cat_onehot:
            transformers.append(
                (
                    "cat_onehot",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_onehot,
                )
            )

        if cat_ordinal:
            transformers.append(
                (
                    "cat_ordinal",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    cat_ordinal,
                )
            )

        return ColumnTransformer(transformers=transformers)
