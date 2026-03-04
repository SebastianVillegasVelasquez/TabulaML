import pandas as pd

from app.core.context.run_context import RunContext, StageResult
from app.core.context.stages import Stages
from app.core.domain.feature_config import FeatureConfig
from app.core.ml.preprocessing_stage import PreprocessingBuilder
from app.core.stages.data_inspection.ordinal_keywords import ORDINAL_KEYWORDS


class DataInspectionStage:

    def __init__(self, context=RunContext):
        self.feature_configs = None
        self.context = context

    def run(self):
        from app.utils.logger import logger
        logger.info("Running data inspection stage...")
        self._inspect_data()

    def _inspect_data(self) -> None:
        """
        Inspects the dataframe and creates a FeatureConfig instance
        for each column.

        :return: None
        """
        df = self.context.config.X_train
        self.num_rows, self.num_cols = df.shape
        self.feature_configs = []

        df = self._drop_redundant_columns(df)

        for col in df.columns:
            series = df[col]

            feature_type = self.detect_feature_type(series)

            cardinality = None
            encoding = None
            skewness = None
            zero_ratio = None
            suggested_transformation = None

            is_categorical = False
            is_numerical = False

            # -------- CATEGORICAL FEATURES --------
            if feature_type == "categorical":
                is_categorical = True
                cardinality = self.compute_cardinality(series)
                encoding = self.decide_categorical_encoding(series, cardinality)

            # -------- NUMERICAL FEATURES --------
            elif feature_type == "numerical":
                is_numerical = True
                dist_info = self.analyze_numerical_distribution(series)
                skewness = dist_info["skewness"]
                zero_ratio = dist_info["zero_ratio"]
                suggested_transformation = dist_info["suggested_transformation"]

            # -------- BINARY FEATURES --------
            elif feature_type == "binary":
                # Binary features are treated as numerical by default
                is_numerical = True

            feature_config = FeatureConfig(
                name=col,
                dtype=str(series.dtype),
                feature_type=feature_type,
                cardinality=cardinality,
                encoding=encoding,
                is_categorical=is_categorical,
                is_numerical=is_numerical,
                skewness=skewness,
                zero_ratio=zero_ratio,
                suggested_transformation=suggested_transformation
            )
            self.feature_configs.append(feature_config)

            self.context.update_context(
                stage=Stages.DATA_HANDLER,
                stage_result=StageResult(
                    name=Stages.DATA_HANDLER.value,
                    results={
                        "preprocessing": PreprocessingBuilder(self.feature_configs).build()
                    }
                )
            )

    @staticmethod
    def _drop_redundant_columns(
            df: pd.DataFrame,
            null_threshold: float = 0.9,
            unique_ratio_threshold: float = 0.90,
    ):
        columns_to_drop = []

        n_rows = len(df)

        for col in df.columns:
            series = df[col]

            # High null ratio
            if series.isnull().mean() > null_threshold:
                columns_to_drop.append(col)
                continue

            # Constant column
            if series.nunique(dropna=False) <= 1:
                columns_to_drop.append(col)
                continue

            # ID-like detection (it is supposed ONLY for object or integer types)
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_integer_dtype(series):
                unique_ratio = series.nunique() / n_rows
                if unique_ratio > unique_ratio_threshold:
                    columns_to_drop.append(col)
                    continue

        return df.drop(columns_to_drop, axis=1)

    @staticmethod
    def detect_feature_type(
            series: pd.Series
    ) -> str:
        """
        Detects the semantic type of feature.

        Possible outputs:
        - 'Binary'
        - 'Categorical'
        - 'Numerical'

        :param series: Pandas Series representing a feature column.
        :return: Detected feature type as a string.
        """
        s = series.dropna()

        # Boolean columns
        if pd.api.types.is_bool_dtype(s):
            return "binary"

        # Binary numeric or object (e.g., 0/1, yes/no)
        if s.nunique() == 2 and pd.api.types.is_object_dtype(s):
            return "binary"

        if pd.api.types.is_numeric_dtype(s) and s.nunique() == 2:
            return "binary"

        # Explicit categorical dtype
        if isinstance(s.dtype, pd.CategoricalDtype):
            return "categorical"

        # Object dtype (strings)
        if pd.api.types.is_object_dtype(s) and s.nunique() > 2:
            return "categorical"

        # Numeric columns
        if pd.api.types.is_numeric_dtype(s):
            return "numerical"

        # Fallback
        return "categorical"

    @staticmethod
    def compute_cardinality(series: pd.Series) -> int:
        """
        Computes the cardinality (number of unique values) of a feature.

        :param series: Pandas Series representing a feature column.
        :return: Number of unique non-null values.
        """
        return series.dropna().nunique()

    @staticmethod
    def suggest_categorical_encoding(
            cardinality: int,
            onehot_max_cardinality: int = 10
    ) -> str:
        """
        Suggests an encoding technique for categorical features.

        :param cardinality: Number of unique values in the feature.
        :param onehot_max_cardinality: Maximum cardinality to apply OneHotEncoding.
        :return: Suggested encoding method ('onehot' or 'ordinal').
        """
        if cardinality <= onehot_max_cardinality:
            return "onehot"
        return "ordinal"

    @staticmethod
    def detect_ordinal_semantics(series: pd.Series) -> bool:
        """
        Detects whether a categorical feature likely represents an ordinal variable
        based on semantic keywords.

        :param series: Pandas Series representing a categorical feature.
        :return: True if ordinal semantics are detected, False otherwise.
        """
        values = (
            series
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )

        matches = 0
        for val in values:
            for keyword in ORDINAL_KEYWORDS:
                if keyword in val:
                    matches += 1
                    break

        # Heuristic: at least 2 ordinal-like values
        return matches >= 2

    @staticmethod
    def decide_categorical_encoding(
            series: pd.Series,
            cardinality: int,
            onehot_max_cardinality: int = 10
    ) -> str:
        """
        Decides the categorical encoding strategy.

        :param series: Pandas Series representing the feature.
        :param cardinality: Number of unique values.
        :param onehot_max_cardinality: Max cardinality for OneHot encoding.
        :return: Encoding strategy ('onehot' or 'ordinal').
        """
        if DataInspectionStage.detect_ordinal_semantics(series=series):
            return "ordinal"

        if cardinality <= onehot_max_cardinality:
            return "onehot"

        return "ordinal"

    @staticmethod
    def analyze_numerical_distribution(series: pd.Series) -> dict:
        """
        Analyzes the distribution of a numerical feature to detect skewness
        and suggest transformations.

        :param series: Pandas Series representing a numerical feature.
        :return: Dictionary with distribution analysis.
        """
        s = series.dropna()

        skewness = s.skew()
        zero_ratio = (s == 0).mean()

        transformation = None

        if abs(skewness) > 1:
            if (s <= 0).any():
                transformation = "yeo-johnson"
            else:
                transformation = "log"

        return {
            "skewness": skewness,
            "zero_ratio": zero_ratio,
            "suggested_transformation": transformation
        }
