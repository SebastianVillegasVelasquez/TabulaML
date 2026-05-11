from collections import defaultdict
from enum import Enum

from sklearn.compose import ColumnTransformer

from app.utils.logger import logger
from .feature_config import FeatureConfig
from .features_container import FeatureContainer
from .feature_config_enum import FeatureType
from .features_container import (
    _build_numerical_pipeline,
    _build_categorical_pipeline,
    _build_boolean_pipeline,
)


class PreprocessingBuilder:
    """Assembles a sklearn ColumnTransformer from typed feature metadata.

    Consumes the list of FeatureConfig subclass instances produced by
    DataInspectionStage and builds one Pipeline per group of features sharing
    the same preprocessing signature. The resulting ColumnTransformer uses
    set_output(transform="pandas") so it returns a DataFrame with the original
    column names rather than a numpy array.

    This class never fits or transforms data — it only constructs the unfitted
    transformer graph that DataInspectionStage will call fit_transform on.

    Attributes:
        feature_configs: Full list of typed feature instances from inspection.
            IDENTIFIER features and features with drop=True are always skipped.
        containers: Populated during build() — one FeatureContainer per model_based.

    Example:
        >>> builder = PreprocessingBuilder(feature_configs)
        >>> column_transformer = builder.build()
        >>> df_preprocessed = column_transformer.fit_transform(X_train)
        >>> df_preprocessed.columns   # human-readable original-style names
    """

    def __init__(self, feature_configs: list[FeatureConfig]) -> None:
        self.feature_configs = feature_configs
        self.containers: list[FeatureContainer] = []

    def build(self) -> ColumnTransformer:
        """Constructs and returns the unfitted ColumnTransformer.

        Orchestrates the full build sequence:
            1. Groups features by FeatureType into FeatureContainer instances.
            2. Dispatches each container to the appropriate pipeline builder.
            3. Assembles all (name, pipeline, cols) tuples into a single
               ColumnTransformer with pandas output and clean column names.

        Returns:
            An unfitted ColumnTransformer ready for fit_transform().

        Raises:
            ValueError: If no valid transformers could be built — usually means
                all features were IDENTIFIER or marked drop=True.
        """
        self._group_features_by_type()

        all_transformers: list[tuple] = []
        for container in self.containers:
            all_transformers.extend(self._dispatch_container(container))

        if not all_transformers:
            raise ValueError(
                "PreprocessingBuilder produced no transformers. "
                "Verify that feature_configs contains at least one non-IDENTIFIER feature."
            )

        column_transformer = ColumnTransformer(
            transformers=all_transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        logger.info(
            "ColumnTransformer assembled: %d transformer group(s).",
            len(all_transformers),
        )
        return column_transformer

    def _group_features_by_type(self) -> None:
        """Groups feature configs by FeatureType into FeatureContainer instances.

        IDENTIFIER features are skipped unconditionally. Features explicitly
        marked with drop=True are also excluded.

        Populates self.containers with one FeatureContainer per detected model_based,
        preserving the order in which types were first encountered.
        """
        grouped: dict[Enum, list[FeatureConfig]] = defaultdict(list)

        for feature in self.feature_configs:
            if feature.feature_type == FeatureType.IDENTIFIER or feature.drop:
                continue
            grouped[feature.feature_type].append(feature)

        self.containers = [
            FeatureContainer(feature_type=ftype, features=features)
            for ftype, features in grouped.items()
        ]

        for container in self.containers:
            logger.debug(
                "Container '%s': %d feature(s) → %s",
                container.feature_type.value,
                len(container.features),
                [f.name for f in container.features],
            )

    @staticmethod
    def _dispatch_container(container: FeatureContainer) -> list[tuple]:
        """Routes a FeatureContainer to the correct pipeline builder function.

        Each FeatureType maps to a dedicated builder that understands the
        metadata specific to that model_based. Unrecognized or unimplemented types
        are logged as warnings and return an empty list, causing those columns
        to be silently dropped by ColumnTransformer(remainder="drop").

        Args:
            container: A FeatureContainer holding features of one FeatureType.

        Returns:
            List of (name, Pipeline, column_names) tuples produced by the
            matching builder, or an empty list for unhandled types.
        """
        f_type = container.feature_type
        features = container.features

        if f_type == FeatureType.NUMERICAL:
            return _build_numerical_pipeline(features)

        if f_type == FeatureType.CATEGORICAL_NOMINAL:
            return _build_categorical_pipeline(nominal=features, ordinal=[])

        if f_type == FeatureType.CATEGORICAL_ORDINAL:
            return _build_categorical_pipeline(nominal=[], ordinal=features)

        if f_type == FeatureType.BOOLEAN:
            return _build_boolean_pipeline(features)

        if f_type in (FeatureType.DATETIME, FeatureType.TEXT):
            logger.warning(
                "FeatureType.%s pipeline is not yet implemented. "
                "Columns %s will be dropped.",
                f_type.name,
                [f.name for f in features],
            )
            return []

        logger.warning(
            "Unhandled FeatureType '%s'. Columns will be dropped.", f_type.value
        )
        return []
