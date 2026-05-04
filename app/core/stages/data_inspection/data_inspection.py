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
from .feature_config_enum import NumericalSubtype
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

        CYCLIC_DOMAINS: Mapping of (min, max, keywords) tuples to the corresponding
        ciclyc domain name. Used to detect periodic columns more easily.

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

    CYCLIC_DOMAINS: list[tuple[int, int, list[str]]] = [
        (0, 23, ["hour", "hora"]),
        (1, 12, ["month", "mes"]),
        (0, 6, ["weekday", "dayofweek", "day_of_week", "dia_semana"]),
        (1, 7, ["weekday", "dayofweek", "day_of_week", "dia_semana"]),
        (1, 31, ["day", "dia"]),
        (1, 4, ["quarter", "trimestre"]),
    ]

    # Fraction of missing values above which a column is dropped outright.
    _NULL_THRESHOLD: float = 0.9

    # Unique-value ratio above which an integer/object column is treated as an ID.
    _ID_UNIQUE_RATIO_THRESHOLD: float = 0.90

    # Minimum average token count to consider a text column as free-form prose.
    _SEMANTIC_TOKEN_THRESHOLD: float = 6.0

    def __init__(self, context: "Context") -> None:

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
        df: pd.DataFrame = self.context.config.dataset.X_train
        self.feature_configs = []

        # Remove structurally uninformative columns and register them as identifiers.
        df, identifier_configs = self._drop_redundant_columns(df)
        self.feature_configs.extend(identifier_configs)

        for col in df.columns:
            series: pd.Series = df[col]
            feature = self._build_feature_config(col, series)
            self.feature_configs.append(feature)

        preprocessing_builder = PreprocessingBuilder(self.feature_configs).build()
        logger.debug(f"Built preprocessing pipeline: {preprocessing_builder}")
        self._handle_update_context(transformer=preprocessing_builder, df=df)

    def _handle_update_context(self, transformer: ColumnTransformer, df: pd.DataFrame) -> None:
        """Fits the transformer and stores the result in the shared context.

        Args:
            transformer: The ColumnTransformer built by PreprocessingBuilder.
            df: The filtered training DataFrame to fit on.
        """
        from app.core.context import StageResult

        # Align the stored DataFrame with the same index reset applied in fitting
        df_clean = df.reset_index(drop=True)

        self._fit_and_log_transformer(transformer=transformer, df=df_clean)

        self.context.update_stage_context(
            stage=Stages.DATA_HANDLER,
            stage_result=StageResult(
                name=Stages.DATA_HANDLER,
                results={
                    "preprocessing": transformer,
                    "df_transformed": self._tranform_df(transformer=transformer, df=df_clean),
                },
            ),
        )

    @staticmethod
    def _tranform_df(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the transformer to the DataFrame and returns the result."""

        # Reset dataframe index to align with the one used in fitting
        df.reset_index()

        # Apply allthe transformation in the original dataframe
        data = transformer.fit_transform(df)

        df_transformed = pd.DataFrame(
            data, columns=transformer.get_feature_names_out(), index=df.index
        )
        return df_transformed

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
            "numerical": self._build_numerical_feature,
            "boolean": self._build_boolean_feature,
            "datetime": self._build_datetime_feature,
            "text": self._build_text_feature,
            "categorical": self._build_categorical_feature,
        }

        builder = builders.get(feature_type, self._build_categorical_feature)
        return builder(col=col, series=series, missing_ratio=missing_ratio)

    # ------------------------------------------------------------------
    # Typed feature builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_numerical_feature(
        col: str,
        series: pd.Series,
        missing_ratio: float,
    ) -> NumericalFeature:
        """Computes raw numerical statistics and instantiates a NumericalFeature.

        Responsibility is intentionally limited to fact-gathering: it computes
        skewness, zero ratio, outlier ratio, variance, and subtype, then hands
        those numbers to NumericalFeature. All decisions about what to *do* with
        those numbers (which scaler, which imputer, which transformation) are
        delegated to the feature's computed properties.

        Args:
            col: Column name as it appears in the dataframe.
            series: Raw pandas Series for the column (nulls included).
            missing_ratio: Pre-computed fraction of null values in [0, 1].

        Returns:
            A NumericalFeature whose properties expose the full preprocessing
            strategy derived from the computed statistics.
        """
        s = series.dropna()

        return NumericalFeature(
            name=col,
            dtype=str(series.dtype),
            skewness=float(s.skew()),
            zero_ratio=float((s == 0).mean()),
            has_negative_values=bool((s < 0).any()),
            outlier_ratio=DataInspectionStage._compute_outlier_ratio(s),
            variance=float(s.var()),
            subtype=DataInspectionStage.infer_numerical_subtype(series, col_name=col),
            missing_ratio=missing_ratio,
        )

    @staticmethod
    def _compute_outlier_ratio(s: pd.Series) -> float:
        """Computes the fraction of values outside the 1.5×IQR fences.

        Uses the Tukey fence method, which is robust to non-normal distributions
        and does not assume any particular distributional shape.

        Args:
            s: Cleaned pandas Series with nulls already dropped.

        Returns:
            Fraction of values classified as outliers, in [0.0, 1.0].
            Returns 0.0 if IQR is zero (constant-like column).
        """
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            # All values in the IQR are identical — no meaningful outlier signal
            return 0.0

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        return float(((s < lower_fence) | (s > upper_fence)).mean())

    @staticmethod
    def _is_effectively_integer(s: pd.Series) -> bool:
        """Checks if all values are whole numbers regardless of the stored dtype.

        Pandas promotes integer columns to float64 when NaNs are present, so
        dtype alone is not a reliable signal. This method checks the actual
        values after nulls are dropped.

        Args:
            s: Cleaned pandas Series with nulls already dropped.

        Returns:
            True if every value in the series is a whole number.
        """
        if pd.api.types.is_integer_dtype(s):
            return True
        if pd.api.types.is_float_dtype(s):
            return bool((s % 1 == 0).all())
        return False

    @staticmethod
    def infer_numerical_subtype(series: pd.Series, col_name: str) -> NumericalSubtype:
        """Infers the numerical subtype using a name-aware heuristic cascade.

        The key improvement over range-only detection is that CYCLIC classification
        now requires the column name to contain a domain keyword, preventing
        low-cardinality count variables (Parch 0-6, SibSp 0-8) from being
        misclassified as periodic just because their range fits a cyclic domain.

        Priority order:
            1. Binary encoded        — values are subset of {0, 1}
            2. Cyclic                — name signals periodicity AND range fits domain
            3. Low-cardinality count — non-negative whole numbers, <=10 distinct values
            4. Count                 — non-negative whole numbers, <=30 distinct values
            5. Ordinal encoded       — whole numbers, small evenly-spaced range
            6. Continuous            — fallback

        Args:
            series: Raw pandas Series for the column (nulls accepted).
            col_name: Column name used for cyclic keyword detection.

        Returns:
            The most appropriate NumericalSubtype for this column.
        """
        s = series.dropna()
        unique_vals = set(s.unique())
        cardinality = len(unique_vals)
        is_int = DataInspectionStage._is_effectively_integer(s)

        # 1. Binary encoded — only 0s and 1s present
        if unique_vals <= {0, 1}:
            return NumericalSubtype.BINARY_ENCODED

        # 2. Cyclic — name + range must both signal periodicity.
        #    Pure range matching is insufficient: Parch (0-6) would match the
        #    "day of week (0-6)" domain even though it is not periodic.
        if is_int and DataInspectionStage._is_cyclic(s, col_name):
            return NumericalSubtype.CYCLIC

        # 3. Low-cardinality count — non-negative integers with very few distinct
        #    values. Each value behaves more like a category than a magnitude
        #    (0 children, 1 child, 2 children). Pipeline should one-hot encode.
        if is_int and (s >= 0).all() and cardinality <= 10:
            return NumericalSubtype.LOW_CARDINALITY_COUNT

        # 4. Count — non-negative integers with moderate cardinality.
        #    The right tail is typically Poisson-distributed; log1p compresses it.
        if is_int and (s >= 0).all() and cardinality <= 30:
            return NumericalSubtype.COUNT

        # 5. Ordinal encoded — small integer range with uniform spacing.
        #    Uniform gaps (1/2/3/4/5 or 0/5/10/15) signal a rating or score scale.
        #    Non-uniform gaps suggest an arbitrary code → falls through to CONTINUOUS.
        if is_int and cardinality <= 15:
            sorted_vals = sorted(unique_vals)
            gaps = {sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)}
            if len(gaps) == 1:
                return NumericalSubtype.ORDINAL_ENCODED

        # 6. Fallback — float or wide-range integer where magnitude is meaningful
        return NumericalSubtype.CONTINUOUS

    @staticmethod
    def _is_cyclic(s: pd.Series, col_name: str) -> bool:
        """Detects whether a column is a periodic/cyclic variable.

        Requires BOTH conditions to be true simultaneously:
            1. The column name contains a keyword associated with a known
               periodic domain (hour, month, weekday, etc.).
            2. The actual value range fits within that domain's bounds.

        This dual requirement prevents low-cardinality count variables
        (Parch, SibSp) from being misclassified as cyclic just because
        their integer range accidentally matches a periodic domain.

        Args:
            s: Cleaned pandas Series with nulls dropped.
            col_name: Original column name used for keyword matching.

        Returns:
            True if both the name and the range signal a cyclic variable.
        """
        col_lower = col_name.lower()
        col_min, col_max = int(s.min()), int(s.max())

        for domain_min, domain_max, keywords in DataInspectionStage.CYCLIC_DOMAINS:
            name_matches = any(kw in col_lower for kw in keywords)
            range_fits = col_min >= domain_min and col_max <= domain_max
            if name_matches and range_fits:
                return True
        return False

    @staticmethod
    def _detect_feature_type(series: pd.Series) -> str:
        """Infers the semantic type of a column.

        Boolean detection reserves the 'boolean' type only for columns whose
        binary values are unambiguously truth-values (True/False, 0/1, yes/no).
        String columns with two arbitrary values like 'male'/'female' are
        classified as 'categorical' so they receive proper encoding instead
        of an unsafe integer cast.

        Args:
            series: Raw pandas Series for the column (nulls included).

        Returns:
            One of: 'boolean', 'datetime', 'text', 'categorical', 'numerical'.
        """
        s = series.dropna()

        # Native bool dtype — unambiguously boolean
        if pd.api.types.is_bool_dtype(s):
            return "boolean"

        # Numeric with exactly two values — treat as boolean (e.g., 0/1 flags)
        if pd.api.types.is_numeric_dtype(s) and s.nunique() == 2:
            return "boolean"

        # Object with exactly two values — only boolean if values are
        # recognised truth-value pairs. Otherwise → categorical.
        if pd.api.types.is_object_dtype(s) and s.nunique() == 2:
            BOOL_PAIRS = {
                frozenset({"true", "false"}),
                frozenset({"yes", "no"}),
                frozenset({"si", "no"}),
                frozenset({"1", "0"}),
                frozenset({"y", "n"}),
            }
            unique_lower = frozenset(s.astype(str).str.lower().str.strip().unique())
            if unique_lower in BOOL_PAIRS:
                return "boolean"
            # 'male'/'female', 'a'/'b', etc. → nominal categorical with cardinality 2
            return "categorical"

        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"

        if pd.api.types.is_object_dtype(s):
            sample = s.head(50)
            try:
                parsed = pd.to_datetime(sample, errors="raise", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    return "datetime"
            except (ValueError, TypeError):
                pass

            avg_tokens = s.astype(str).str.split().str.len().mean()
            if avg_tokens >= 6:
                return "text"

            return "categorical"

        if isinstance(s.dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_numeric_dtype(s):
            return "numerical"

        return "categorical"

    @staticmethod
    def _build_boolean_feature(col: str, series: pd.Series, missing_ratio: float) -> BooleanFeature:
        """Computes binary statistics and instantiates a BooleanFeature.

        By the time this builder is called, _detect_feature_type has already
        confirmed the column is a recognised boolean (native bool, 0/1 numeric,
        or a known truth-value string pair). This builder only computes the
        true_ratio statistic and records the dtype for pipeline routing.

        Args:
            col: Column name.
            series: Raw pandas Series for the column (nulls included).
            missing_ratio: Pre-computed proportion of null values.

        Returns:
            A BooleanFeature with true_ratio computed from the actual values.
        """
        s = series.dropna()

        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            # Numeric/bool: True and the larger of {0,1} both cast safely to bool
            true_ratio = float(s.astype(bool).mean())
        else:
            # String boolean pair recognised by _detect_feature_type
            # The positive class is whichever value maps to True in common usage.
            # We use the less frequent value as a proxy (minority = positive class).
            positive_proxies = {"true", "yes", "si", "1", "y"}
            true_ratio = float(s.astype(str).str.lower().str.strip().isin(positive_proxies).mean())

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

    def _build_text_feature(self, col: str, series: pd.Series, missing_ratio: float) -> TextFeature:
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
            1 for val in values if any(keyword in val for keyword in cls.ORDINAL_KEYWORDS)
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

    @staticmethod
    def _fit_and_log_transformer(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the ColumnTransformer and logs the resulting feature names.

        Resets the DataFrame index before fitting to prevent the index-mismatch
        error that occurs when ColumnTransformer concatenates sub-pipeline outputs.
        This happens because some sklearn transformers (e.g. OneHotEncoder with
        set_output="pandas") create a new RangeIndex internally, while others
        preserve the original index. If the input has a non-default index (e.g.
        after train_test_split), the pd.concat inside ColumnTransformer fails.

        Resetting to a clean RangeIndex guarantees all sub-pipeline outputs share
        the same index and can be concatenated without conflicts.

        Args:
            transformer: The unfitted ColumnTransformer assembled by PreprocessingBuilder.
            df: The filtered training DataFrame to fit on.

        Returns:
            The transformed DataFrame with clean column names.
        """
        # Reset index so all sub-pipeline outputs share index 0..N-1.
        # drop=True discards the old index instead of adding it as a column.
        df_clean = df.reset_index(drop=True)

        x_transformed = transformer.fit_transform(df_clean)
        feature_names = list(transformer.get_feature_names_out())

        logger.info("Preprocessed feature names: %s", feature_names)
        logger.debug(
            "Preprocessed DataFrame head:\n%s",
            x_transformed.head() if hasattr(x_transformed, "head") else x_transformed[:5],
        )
        return x_transformed
