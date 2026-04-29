from __future__ import annotations

"""
Data inspection stage for the AutoML pipeline.

This module is responsible for the first stage of the AutoML system: scanning
each column of the input dataset, computing its raw statistics, and mapping it
to the appropriate feature class from the domain model. The resulting list of
typed feature configs is then consumed by the preprocessing pipeline builder.

Typical usage example:

    stage = DataInspectionStage(context)
    stage.run()
    # context now holds the fitted ColumnTransformer and the feature registry
"""
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer

from app.core.enums import Stages
from .feature_config import (
    BooleanFeature,
    CategoricalNominalFeature,
    CategoricalOrdinalFeature,
    DatetimeFeature,
    DatetimeGranularity,
    FeatureConfig,
    IdentifierFeature,
    NumericalFeature,
    TextFeature,
)
from .feature_config_info import NumericalSubtype
from .preprocessing_stage import PreprocessingBuilder
from app.utils.logger import logger


class DataInspectionStage:
    """First stage of the AutoML pipeline: dataset scanning and feature typing.

    Iterates over every column in the training set, computes its raw descriptive
    statistics, and instantiates the corresponding typed feature class
    (NumericalFeature, CategoricalNominalFeature, etc.). The business logic for
    selecting transformations and encodings lives entirely inside each feature
    class via computed properties — this stage is only responsible for gathering
    the raw numbers those properties depend on.

    Attributes:
        ORDINAL_KEYWORDS: Mapping of lowercase semantic keywords to their implied
            ordinal rank. Used as a heuristic to detect ordinal categorical columns
            without explicit dtype information.
        context: Shared AutoML context carrying the dataset and stage results.
        feature_configs: List of typed feature instances built during inspection.
            Populated after ``run()`` is called.

    Example:
        >>> stage = DataInspectionStage(context)
        >>> stage.run()
        >>> stage.feature_configs  # list of NumericalFeature, CategoricalFeature, etc.
    """

    ORDINAL_KEYWORDS: dict[str, int] = {
        "very low": -1,
        "low": 0,
        "small": 0,
        "near": 0,
        "poor": 0,
        "fair": 1,
        "medium": 1,
        "good": 2,
        "large": 2,
        "far": 2,
        "high": 2,
        "very high": 3,
        "excellent": 3,
    }

    # Fraction of missing values above which a column is dropped outright.
    _NULL_THRESHOLD: float = 0.9

    # Unique-value ratio above which an integer/object column is treated as an ID.
    _ID_UNIQUE_RATIO_THRESHOLD: float = 0.90

    # Minimum average token count to consider a text column as free-form prose.
    _SEMANTIC_TOKEN_THRESHOLD: float = 6.0

    def __init__(self, context: "Context") -> None:
        from app.core.context import Context
        self.context = context
        self.feature_configs: list[FeatureConfig] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Executes the data inspection stage.

        Scans the training dataframe, builds the typed feature registry, and
        stores the fitted preprocessing transformer back into the shared context.
        """
        logger.info("Running data inspection stage...")
        self._inspect_data()

    # ------------------------------------------------------------------
    # Core inspection logic
    # ------------------------------------------------------------------

    def _inspect_data(self) -> None:
        """Orchestrates column scanning and context update.

        For each column in the training set this method:
            1. Detects its semantic type (numerical, categorical, boolean, etc.).
            2. Computes the raw statistics required by that feature class.
            3. Instantiates the corresponding typed FeatureConfig subclass.
            4. Delegates transformation/encoding decisions to the feature's properties.

        The resulting feature_configs list is handed off to the PreprocessingBuilder.
        """
        df: pd.DataFrame = self.context.config.X_train()
        self.feature_configs = []

        # Remove structurally uninformative columns and register them as identifiers.
        df, identifier_configs = self._drop_redundant_columns(df)
        self.feature_configs.extend(identifier_configs)

        for col in df.columns:
            series: pd.Series = df[col]
            feature = self._build_feature_config(col, series)
            logger.debug(f"Feature config built: {feature}" )
            self.feature_configs.append(feature)

        logger.debug(f"Feature configs: {self.feature_configs}", )

        preprocessing_builder = PreprocessingBuilder(self.feature_configs).build()
        self._handle_update_context(transformer=preprocessing_builder, df=df)

    def _build_feature_config(self, col: str, series: pd.Series) -> FeatureConfig:
        """Detects the feature type and instantiates the correct typed class.

        Acts as a dispatch method: it calls the appropriate private builder
        based on the detected semantic type and returns the populated feature
        instance. All transformation/encoding decisions are deferred to the
        computed properties of each class.

        Args:
            col: Column name as it appears in the dataframe.
            series: The pandas Series for that column.

        Returns:
            A fully populated FeatureConfig subclass instance whose properties
            reflect the most appropriate preprocessing strategy for this column.
        """
        missing_ratio = float(series.isnull().mean())
        feature_type = self._detect_feature_type(series)

        builders: dict[str, Any] = {
            "numerical":  self._build_numerical_feature,
            "boolean":    self._build_boolean_feature,
            "datetime":   self._build_datetime_feature,
            "text":       self._build_text_feature,
            "categorical": self._build_categorical_feature,
        }

        builder = builders.get(feature_type, self._build_categorical_feature)
        return builder(col=col, series=series, missing_ratio=missing_ratio)

    # ------------------------------------------------------------------
    # Typed feature builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_numerical_feature(
        col: str, series: pd.Series, missing_ratio: float
    ) -> NumericalFeature:
        """Computes numerical statistics and instantiates a NumericalFeature.

        Gathers skewness, zero ratio, outlier ratio, variance, and sign
        information. The suggested transformation is derived automatically
        by the NumericalFeature.suggested_transformation property.

        Args:
            col: Column name.
            series: Raw pandas Series for the column.
            missing_ratio: Pre-computed proportion of null values.

        Returns:
            A NumericalFeature populated with the column's distribution stats.
        """
        s = series.dropna()

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outlier_ratio = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())

        return NumericalFeature(
            name=col,
            dtype=str(series.dtype),
            skewness=float(s.skew()),
            zero_ratio=float((s == 0).mean()),
            has_negative_values=bool((s < 0).any()),
            outlier_ratio=outlier_ratio,
            variance=float(s.var()),
            is_discrete=pd.api.types.is_integer_dtype(s),
            missing_ratio=missing_ratio,
        )

    @staticmethod
    def infer_numerical_subtype(series: pd.Series) -> NumericalSubtype:
        """Infers the numerical subtype from the column's value distribution.

        Applies a priority-based heuristic cascade. Each check is ordered from
        most structurally constrained (binary, cyclic) to least (continuous),
        so that edge cases are caught before the general case.

        Priority order:
            1. Binary encoded  — exactly {0, 1}
            2. Cyclic          — integer, range matches known periodic domains
            3. Count           — non-negative integer, low cardinality
            4. Ordinal encoded — small-range integer, evenly spaced values
            5. Continuous      — fallback for floats and wide-range integers

        Args:
            series: The raw pandas Series for the column (nulls allowed).

        Returns:
            The most appropriate NumericalSubtype for this column.
        """
        s = series.dropna()

        unique_vals = set(s.unique())
        cardinality = len(unique_vals)
        is_integer = pd.api.types.is_integer_dtype(s)

        # 1. Binary encoded: only values are 0 and 1
        if unique_vals <= {0, 1}:
            return NumericalSubtype.BINARY_ENCODED

        # 2. Cyclic: integer column whose range matches a known periodic domain
        #    Common domains: hour (0-23), month (1-12), day of week (0-6 or 1-7),
        #    day of month (1-31), quarter (1-4).
        CYCLIC_DOMAINS = [
            (0, 23),  # hour
            (1, 12),  # month
            (0, 6),  # day of week (Python convention)
            (1, 7),  # day of week (ISO convention)
            (1, 31),  # day of month
            (1, 4),  # quarter
        ]
        if is_integer:
            col_min, col_max = int(s.min()), int(s.max())
            for domain_min, domain_max in CYCLIC_DOMAINS:
                if col_min >= domain_min and col_max <= domain_max:
                    return NumericalSubtype.CYCLIC

        # 3. Count: non-negative integer with low cardinality
        #    Threshold of 100 covers practical count features (n_purchases, n_clicks)
        #    without misclassifying encoded IDs.
        if is_integer and (s >= 0).all() and cardinality <= 100:
            return NumericalSubtype.COUNT

        # 4. Ordinal encoded: small integer range, evenly spaced (e.g., 1/2/3/4/5)
        #    Evenly spaced check avoids confusing arbitrary codes with ordinal scales.
        if is_integer and cardinality <= 15:
            sorted_vals = sorted(unique_vals)
            gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
            if gaps and len(set(gaps)) == 1:  # all gaps equal → evenly spaced
                return NumericalSubtype.ORDINAL_ENCODED

        # 5. Fallback: continuous
        return NumericalSubtype.CONTINUOUS

    @staticmethod
    def _build_boolean_feature(
        col: str, series: pd.Series, missing_ratio: float
    ) -> BooleanFeature:
        """Computes binary statistics and instantiates a BooleanFeature.

        Args:
            col: Column name.
            series: Raw pandas Series for the column.
            missing_ratio: Pre-computed proportion of null values.

        Returns:
            A BooleanFeature with the proportion of True/1 values recorded.
        """
        s = series.dropna()

        # Normalize heterogeneous binary representations to boolean
        if pd.api.types.is_object_dtype(s):
            positive_values = {"yes", "true", "1", "y"}
            true_ratio = float(s.astype(str).str.lower().str.strip().isin(positive_values).mean())
        else:
            true_ratio = float(s.astype(bool).mean())

        return BooleanFeature(
            name=col,
            dtype=str(series.dtype),
            true_ratio=true_ratio,
            missing_ratio=missing_ratio,
        )

    def _build_categorical_feature(
        self, col: str, series: pd.Series, missing_ratio: float
    ) -> CategoricalNominalFeature | CategoricalOrdinalFeature:
        """Computes categorical statistics and dispatches to ordinal or nominal.

        Applies keyword-based heuristics to detect ordinal semantics. If ordinal
        patterns are found, a CategoricalOrdinalFeature is returned; otherwise
        a CategoricalNominalFeature is returned. Encoding decisions are left
        entirely to each class's suggested_encoding property.

        Args:
            col: Column name.
            series: Raw pandas Series for the column.
            missing_ratio: Pre-computed proportion of null values.

        Returns:
            A CategoricalOrdinalFeature if ordinal semantics are detected,
            otherwise a CategoricalNominalFeature.
        """
        s = series.dropna()
        cardinality = int(s.nunique())
        value_counts = s.value_counts(normalize=True)
        most_frequent_ratio = float(value_counts.iloc[0]) if not value_counts.empty else 0.0
        has_rare = bool((value_counts < 0.01).any())

        common_kwargs = dict(
            name=col,
            dtype=str(series.dtype),
            cardinality=cardinality,
            most_frequent_ratio=most_frequent_ratio,
            has_rare_categories=has_rare,
            missing_ratio=missing_ratio,
        )

        if self._detect_ordinal_semantics(series):
            category_order = self._infer_ordinal_order(series)
            return CategoricalOrdinalFeature(**common_kwargs, category_order=category_order)

        return CategoricalNominalFeature(**common_kwargs)

    @staticmethod
    def _build_datetime_feature(
        col: str, series: pd.Series, missing_ratio: float
    ) -> DatetimeFeature:
        """Computes temporal metadata and instantiates a DatetimeFeature.

        Args:
            col: Column name.
            series: Raw pandas Series for the column.
            missing_ratio: Pre-computed proportion of null values.

        Returns:
            A DatetimeFeature with granularity, date range, and timezone info.
        """
        s = pd.to_datetime(series, errors="coerce").dropna()

        has_time = bool((s.dt.hour != 0).any() or (s.dt.minute != 0).any())
        granularity = DatetimeGranularity.DATETIME if has_time else DatetimeGranularity.DATE

        return DatetimeFeature(
            name=col,
            dtype=str(series.dtype),
            granularity=granularity,
            min_date=str(s.min().date()) if not s.empty else None,
            max_date=str(s.max().date()) if not s.empty else None,
            has_timezone=bool(s.dt.tz is not None),
            missing_ratio=missing_ratio,
        )

    def _build_text_feature(
        self, col: str, series: pd.Series, missing_ratio: float
    ) -> TextFeature:
        """Computes text corpus statistics and instantiates a TextFeature.

        Args:
            col: Column name.
            series: Raw pandas Series for the column.
            missing_ratio: Pre-computed proportion of null values.

        Returns:
            A TextFeature with vocabulary size, average token count, and
            semantic flag. The suggested vectorization is derived automatically
            by the TextFeature.suggested_vectorization property.
        """
        s = series.dropna().astype(str)
        token_counts = s.str.split().str.len()
        avg_tokens = float(token_counts.mean())
        vocabulary = set(word for text in s for word in text.lower().split())

        return TextFeature(
            name=col,
            dtype=str(series.dtype),
            avg_token_count=avg_tokens,
            vocabulary_size=len(vocabulary),
            is_semantic=avg_tokens >= self._SEMANTIC_TOKEN_THRESHOLD,
            missing_ratio=missing_ratio,
        )

    # ------------------------------------------------------------------
    # Column filtering
    # ------------------------------------------------------------------

    def _drop_redundant_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[IdentifierFeature]]:
        """Removes structurally uninformative columns from the dataframe.

        A column is considered redundant if it meets any of the following:
            - Its null ratio exceeds ``_NULL_THRESHOLD``.
            - It is constant (cardinality <= 1).
            - Its unique-value ratio exceeds ``_ID_UNIQUE_RATIO_THRESHOLD``
              for object or integer dtypes, indicating an ID-like column.

        Dropped columns are registered as IdentifierFeature instances so the
        pipeline builder is aware of them and can skip them explicitly.

        Args:
            df: The raw training dataframe.

        Returns:
            A tuple of (filtered_dataframe, list_of_identifier_feature_configs).
        """
        columns_to_drop: list[str] = []
        identifier_configs: list[IdentifierFeature] = []
        n_rows = len(df)

        for col in df.columns:
            series = df[col]

            if series.isnull().mean() > self._NULL_THRESHOLD:
                columns_to_drop.append(col)
                identifier_configs.append(self._make_identifier(col, series, n_rows))
                logger.debug("Dropping '%s': exceeds null threshold.", col)
                continue

            if series.nunique(dropna=False) <= 1:
                columns_to_drop.append(col)
                identifier_configs.append(self._make_identifier(col, series, n_rows))
                logger.debug("Dropping '%s': constant column.", col)
                continue

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_integer_dtype(series):
                unique_ratio = series.nunique() / n_rows
                if unique_ratio > self._ID_UNIQUE_RATIO_THRESHOLD:
                    columns_to_drop.append(col)
                    identifier_configs.append(self._make_identifier(col, series, n_rows))
                    logger.debug("Dropping '%s': ID-like unique ratio %.2f.", col, unique_ratio)
                    continue

        return df.drop(columns=columns_to_drop), identifier_configs

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_feature_type(series: pd.Series) -> str:
        """Infers the semantic type of a column from its dtype and value set.

        Detection priority:
            1. Boolean dtype → ``'boolean'``
            2. Two-value object or numeric → ``'boolean'``
            3. Datetime-parseable object → ``'datetime'``
            4. Long free-text strings → ``'text'``
            5. Explicit CategoricalDtype or object with > 2 unique values → ``'categorical'``
            6. Numeric → ``'numerical'``
            7. Fallback → ``'categorical'``

        Args:
            series: The pandas Series to evaluate.

        Returns:
            One of: ``'boolean'``, ``'datetime'``, ``'text'``,
            ``'categorical'``, or ``'numerical'``.
        """
        s = series.dropna()

        if pd.api.types.is_bool_dtype(s):
            return "boolean"

        if s.nunique() == 2:
            return "boolean"

        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"

        # Attempt datetime parsing for object columns before classifying as text
        if pd.api.types.is_object_dtype(s):
            sample = s.head(50)
            try:
                parsed = pd.to_datetime(sample, errors="raise", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    return "datetime"
            except (ValueError, TypeError):
                pass

            # Detect free-text columns by average word count
            avg_tokens = s.astype(str).str.split().str.len().mean()
            if avg_tokens >= 6:
                return "text"

            return "categorical"

        if isinstance(s.dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_numeric_dtype(s):
            return "numerical"

        return "categorical"

    @classmethod
    def _detect_ordinal_semantics(cls, series: pd.Series) -> bool:
        """Heuristically detects whether a categorical column is ordinal.

        Checks whether at least two unique values in the column match a keyword
        from ``ORDINAL_KEYWORDS``. A match is a substring match, so 'very high
        risk' would match 'very high'.

        Args:
            series: The categorical pandas Series to evaluate.

        Returns:
            True if two or more ordinal-like values are found, False otherwise.
        """
        values = series.dropna().astype(str).str.lower().str.strip().unique()
        matches = sum(
            1 for val in values
            if any(keyword in val for keyword in cls.ORDINAL_KEYWORDS)
        )
        return matches >= 2

    @classmethod
    def _infer_ordinal_order(cls, series: pd.Series) -> list[str] | None:
        """Builds a sorted category list based on ORDINAL_KEYWORDS ranks.

        Only categories that match a keyword are included. Unmatched values
        are silently excluded; if fewer than two values match, returns None
        so the pipeline can fall back to lexicographic ordering.

        Args:
            series: The categorical pandas Series.

        Returns:
            A list of category strings sorted by their ordinal rank, or None
            if not enough matches were found to establish a reliable order.
        """
        values = series.dropna().astype(str).str.lower().str.strip().unique()
        ranked: list[tuple[int, str]] = []

        for val in values:
            for keyword, rank in cls.ORDINAL_KEYWORDS.items():
                if keyword in val:
                    ranked.append((rank, val))
                    break

        if len(ranked) < 2:
            return None

        return [v for _, v in sorted(ranked, key=lambda x: x[0])]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_identifier(col: str, series: pd.Series, n_rows: int) -> IdentifierFeature:
        """Builds an IdentifierFeature for a column being dropped.

        Args:
            col: Column name.
            series: The pandas Series for that column.
            n_rows: Total number of rows in the dataframe.

        Returns:
            An IdentifierFeature with drop=True and cardinality recorded.
        """
        cardinality = int(series.nunique(dropna=False))
        return IdentifierFeature(
            name=col,
            dtype=str(series.dtype),
            cardinality=max(cardinality, 1),
            is_primary_key=cardinality == n_rows,
        )

    def _handle_update_context(
        self, transformer: ColumnTransformer, df: pd.DataFrame
    ) -> None:
        """Fits the transformer and stores the result in the shared context.

        Args:
            transformer: The ColumnTransformer built by PreprocessingBuilder.
            df: The filtered training dataframe to fit on.
        """
        from app.core.context import StageResult

        self._fit_and_log_transformer(transformer=transformer, df=df)

        self.context.update_stage_context(
            stage=Stages.DATA_HANDLER,
            stage_result=StageResult(
                name=Stages.DATA_HANDLER,
                results={"preprocessing": transformer},
            ),
        )

    @staticmethod
    def _fit_and_log_transformer(
        transformer: ColumnTransformer, df: pd.DataFrame
    ) -> None:
        """Fits the ColumnTransformer and logs the resulting feature names.

        Args:
            transformer: The unfitted ColumnTransformer.
            df: The dataframe to fit on.
        """
        x_transformed = transformer.fit_transform(df)
        feature_names = list(transformer.get_feature_names_out())
        logger.info("Preprocessed feature names: %s", feature_names)
        logger.debug(
            "Preprocessed dataframe head:\n%s",
            pd.DataFrame(x_transformed, columns=feature_names).head(),
        )

