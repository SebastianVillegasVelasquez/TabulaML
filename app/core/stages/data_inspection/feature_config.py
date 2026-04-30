from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .feature_config_enum import (
    ImputationStrategy,
    EncodingType,
    TransformationType,
    TextVectorizationType,
    DatetimeGranularity,
    FeatureType,
    NumericalSubtype,
    ScalerStrategy,
)


class FeatureConfig(BaseModel):
    """Base class for feature configuration and characterization.

    Centralizes the universal attributes present in any column type detected
    during exploratory dataset analysis. Serves as the base contract for
    pipeline construction in the AutoML system.

    Attributes:
        name: Original column name in the dataset.
        dtype: Detected pandas/numpy data type (e.g., 'float64', 'object').
        feature_type: Semantic category of the feature (e.g., 'numerical', 'categorical').
        missing_ratio: Proportion of null values in the range [0.0, 1.0].
        is_target: Indicates whether this column is the model's target variable.
        drop: If True, the pipeline must exclude this feature from training.
        notes: Optional additional observations for internal documentation.

    Raises:
        ValueError: If missing_ratio is not within the range [0.0, 1.0].
    """

    model_config = {"frozen": False, "validate_assignment": True}

    name: str = Field(..., description="Original column name.")
    dtype: str = Field(..., description="Detected pandas/numpy dtype.")
    feature_type: FeatureType = Field(..., description="Semantic type of the feature.")
    missing_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of missing values [0, 1].",
    )
    is_target: bool = Field(default=False, description="True if this is the target variable.")
    drop: bool = Field(
        default=False,
        description="True if this feature should be excluded from the pipeline.",
    )
    notes: Optional[str] = Field(default=None, description="Optional internal notes.")

    @property
    def needs_imputation(self) -> bool:
        """Determines whether the feature requires missing value imputation.

        Returns:
            True if the proportion of missing values is greater than zero.
        """
        return self.missing_ratio > 0.0

    @property
    def is_high_missing(self) -> bool:
        """Evaluates whether the missingness rate is critical (> 40%).

        Features with a high proportion of nulls may need to be dropped or
        handled with special strategies (e.g., a binary missingness indicator).

        Returns:
            True if missing_ratio exceeds the 60% critical threshold.
        """
        return self.missing_ratio > 0.6


# ---------------------------------------------------------------------------
# Numerical features
# ---------------------------------------------------------------------------


class NumericalFeature(FeatureConfig):
    """Characterization of a continuous or discrete numerical column.

    Stores the raw descriptive statistics gathered during data inspection.
    All preprocessing decisions (transformation, scaling, imputation) are
    derived lazily from those stats via computed properties, so they always
    reflect the current state of the feature without manual synchronization.

    Design contract:
        - Fields  → raw numbers computed from the data (skewness, ratios, etc.)
        - Properties → decisions derived from those numbers (what to do with them)
        - No decision logic lives in the builder; the builder only fills fields.

    The decision hierarchy shared across the three main properties is:

        Subtype check first (structural constraints override everything):
            BINARY_ENCODED      → no transform, no scale, mean impute
            CYCLIC              → no transform, no scale, median impute
            ORDINAL_ENCODED     → no transform, no scale, median impute
            LOW_CARDINALITY_COUNT → no transform, no scale, median impute
            COUNT               → log1p/log transform, robust scale, median impute

        Distribution check second (for CONTINUOUS only):
            outliers or skewed  → robust scale,   median impute
            clean               → standard scale,  mean impute

    Attributes:
        feature_type: Always FeatureType.NUMERICAL.
        subtype: Semantic subtype (CONTINUOUS, COUNT, CYCLIC, etc.) inferred
            during inspection. Governs the entire decision hierarchy.
        skewness: Pearson skewness coefficient of the non-null values.
            Positive = right tail, negative = left tail.
        zero_ratio: Fraction of non-null values that are exactly zero [0, 1].
        has_negative_values: True if any non-null value is strictly negative.
        outlier_ratio: Fraction of values outside 1.5×IQR fences [0, 1].
        variance: Variance of the non-null values. None if not computed.
        missing_ratio: Fraction of null values in the column [0, 1].

    Raises:
        ValueError: If zero_ratio, outlier_ratio, or missing_ratio fall
            outside [0, 1].
    """

    feature_type: FeatureType = Field(default=FeatureType.NUMERICAL)
    subtype: NumericalSubtype = Field(
        default=NumericalSubtype.CONTINUOUS,
        description="Semantic subtype governing the full preprocessing decision hierarchy.",
    )
    skewness: float = Field(..., description="Distribution skewness coefficient.")
    zero_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Proportion of non-null values equal to zero.",
    )
    has_negative_values: bool = Field(
        default=False,
        description="True if any non-null value is strictly negative.",
    )
    outlier_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Proportion of values outside the 1.5×IQR fences.",
    )
    variance: Optional[float] = Field(
        default=None,
        description="Variance of non-null values. None if not computed.",
    )

    @property
    def is_skewed(self) -> bool:
        """Returns True if |skewness| exceeds the conventional threshold of 1.0.

        Returns:
            True when the distribution has significant asymmetry.
        """
        return abs(self.skewness) > 1.0

    @property
    def is_near_zero_variance(self) -> bool:
        """Returns True if the feature carries almost no information.

        Near-zero variance features are typically safe to drop before
        fitting any estimator.

        Returns:
            True if variance < 1e-6, or False when variance is unavailable.
        """
        if self.variance is None:
            return False
        return self.variance < 1e-6

    @property
    def has_outliers(self) -> bool:
        """Returns True if the outlier fraction exceeds the 5% threshold.

        Returns:
            True when more than 5% of values are outside the IQR fences.
        """
        return self.outlier_ratio > 0.05

    @property
    def which_encoding_needs(self):
        """Returns True if the feature should be converted to categorical."""
        from .feature_config_enum import EncodingType
        match self.subtype:
            case NumericalSubtype.BINARY_ENCODED:
                return EncodingType.BINARY
            case NumericalSubtype.CYCLIC:
                return EncodingType.FREQUENCY
            case NumericalSubtype.ORDINAL_ENCODED:
                return EncodingType.ORDINAL
            case NumericalSubtype.LOW_CARDINALITY_COUNT:
                return EncodingType.ONEHOT
            case _:
                return None


    @property
    def suggested_transformation(self) -> TransformationType:
        """Selects the distribution transformation based on subtype and stats.

        Subtype-driven rules are evaluated first because structural constraints
        (e.g., a cyclic variable must not be log-transformed) are harder
        constraints than distributional ones.

        Decision tree:
            BINARY_ENCODED        → NONE
            CYCLIC                → NONE  (sin/cos encoding handled separately)
            ORDINAL_ENCODED       → NONE  (scaling would destroy ordinal meaning)
            LOW_CARDINALITY_COUNT → NONE  (pipeline will one-hot encode)
            COUNT, zeros present  → LOG1P (handles zeros safely)
            COUNT, no zeros       → LOG
            CONTINUOUS, negatives → YEO_JOHNSON
            CONTINUOUS, normal    → NONE  (scaler is enough)
            CONTINUOUS, skewed + zeros → LOG1P
            CONTINUOUS, skewed, no zeros, positive skew → LOG
            CONTINUOUS, skewed, no zeros, negative skew → YEO_JOHNSON

        Returns:
            The TransformationType that should be applied before scaling.
        """
        # Subtypes that require no distributional transformation
        _no_transform = {
            NumericalSubtype.BINARY_ENCODED,
            NumericalSubtype.CYCLIC,
            NumericalSubtype.ORDINAL_ENCODED,
            NumericalSubtype.LOW_CARDINALITY_COUNT,
        }
        if self.subtype in _no_transform:
            return TransformationType.NONE

        # Count data: Poisson-distributed right tail, log1p is the natural fix
        if self.subtype == NumericalSubtype.COUNT:
            return TransformationType.LOG1P if self.zero_ratio > 0 else TransformationType.LOG

        # CONTINUOUS from here — distribution-based logic
        if self.has_negative_values:
            # Only Yeo-Johnson supports the full real line
            return TransformationType.YEO_JOHNSON

        if not self.is_skewed:
            # Near-normal: no transformation needed; scaler is sufficient
            return TransformationType.NONE

        # Skewed and non-negative
        if self.zero_ratio > 0.0:
            return TransformationType.LOG1P

        return TransformationType.LOG if self.skewness > 1.0 else TransformationType.YEO_JOHNSON

    @property
    def suggested_scaler(self) -> Optional[ScalerStrategy]:
        """Selects the scaling strategy based on subtype and distribution shape.

        Subtypes that will be encoded (not scaled) by the pipeline return None
        so the pipeline builder knows to route them to an encoder instead.

        Decision tree:
            BINARY_ENCODED        → None  (already 0/1, no scaling needed)
            CYCLIC                → None  (will be sin/cos encoded)
            ORDINAL_ENCODED       → None  (will be ordinal encoded)
            LOW_CARDINALITY_COUNT → None  (will be one-hot encoded)
            COUNT                 → ROBUST_SCALER (after log transform)
            CONTINUOUS, outliers or skewed → ROBUST_SCALER
            CONTINUOUS, clean     → STANDARD_SCALER

        Returns:
            The ScalerStrategy to apply after transformation, or None if the
            feature should be encoded rather than scaled.
        """
        _no_scale = {
            NumericalSubtype.BINARY_ENCODED,
            NumericalSubtype.CYCLIC,
            NumericalSubtype.ORDINAL_ENCODED,
            NumericalSubtype.LOW_CARDINALITY_COUNT,
        }
        if self.subtype in _no_scale:
            return None

        # Count data after log1p still benefits from robust scaling
        if self.subtype == NumericalSubtype.COUNT:
            return ScalerStrategy.ROBUST_SCALER

        # CONTINUOUS: outliers or skewness → median-based robust scaler
        if self.has_outliers or self.is_skewed:
            return ScalerStrategy.ROBUST_SCALER

        return ScalerStrategy.STANDARD_SCALER

    @property
    def suggested_imputer(self) -> ImputationStrategy:
        """Selects the imputation strategy based on subtype and distribution shape.

        Mean imputation is only safe when the distribution is clean (no outliers,
        no skewness), because the mean is not a robust statistic. In all other
        cases the median is preferred.

        Decision tree:
            BINARY_ENCODED        → MODE_IMPUTER   (preserve binary semantics)
            CYCLIC                → MEDIAN_IMPUTER
            ORDINAL_ENCODED       → MEDIAN_IMPUTER
            LOW_CARDINALITY_COUNT → MEDIAN_IMPUTER (mode could also work)
            COUNT                 → MEDIAN_IMPUTER (Poisson right-tail skews mean)
            CONTINUOUS, clean     → MEAN_IMPUTER
            CONTINUOUS, dirty     → MEDIAN_IMPUTER

        Returns:
            The ImputationStrategy recommended for this feature.
        """
        if self.subtype == NumericalSubtype.BINARY_ENCODED:
            return ImputationStrategy.MODE

        _median_subtypes = {
            NumericalSubtype.CYCLIC,
            NumericalSubtype.ORDINAL_ENCODED,
            NumericalSubtype.LOW_CARDINALITY_COUNT,
            NumericalSubtype.COUNT,
        }
        if self.subtype in _median_subtypes:
            return ImputationStrategy.MEDIAN

        # CONTINUOUS: only use mean when the distribution is genuinely clean
        if not self.has_outliers and not self.is_skewed:
            return ImputationStrategy.MEAN

        return ImputationStrategy.MEDIAN

    @property
    def preprocessing_summary(self) -> dict:
        """Returns a human-readable summary of all preprocessing decisions.

        Useful for logging and pipeline introspection without having to call
        each property individually.

        Returns:
            Dictionary with keys: subtype, transformation, scaler, imputer,
            is_skewed, has_outliers, needs_imputation.
        """
        return {
            "subtype": self.subtype.value,
            "transformation": self.suggested_transformation.value,
            "scaler": self.suggested_scaler.value if self.suggested_scaler else None,
            "imputer": self.suggested_imputer.value,
            "is_skewed": self.is_skewed,
            "has_outliers": self.has_outliers,
            "needs_imputation": self.needs_imputation,
        }


# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------


class CategoricalFeature(FeatureConfig):
    """Base characterization of a categorical column.

    Provides the fundamental statistical attributes for features that represent
    discrete categories, regardless of whether they have an intrinsic order.

    Attributes:
        cardinality: Number of unique categories detected in the dataset.
        most_frequent_ratio: Proportion of the most frequent category [0, 1].
            Values close to 1.0 indicate near-constant features.
        has_rare_categories: True if any category has frequency < 1%.
            These can cause issues with supervised encodings.
        suggested_imputation: Recommended imputation strategy (default: MODE).

    Raises:
        ValueError: If cardinality is less than or equal to zero.
        ValueError: If most_frequent_ratio is not in [0, 1].
    """

    feature_type: FeatureType = Field(default=FeatureType.CATEGORICAL)
    cardinality: int = Field(..., gt=0, description="Number of unique categories.")
    most_frequent_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of the dominant category.",
    )
    has_rare_categories: bool = Field(
        default=False,
        description="True if any category has frequency < 1%.",
    )
    suggested_imputation: ImputationStrategy = Field(
        default=ImputationStrategy.MODE,
        description="Recommended imputation strategy.",
    )

    @property
    def is_high_cardinality(self) -> bool:
        """Detects high cardinality that discourages One-Hot Encoding.

        The threshold of 20 categories is conventional and may be adjusted
        based on the target model and available memory.

        Returns:
            True if the number of unique categories exceeds 20.
        """
        return self.cardinality > 20

    @property
    def is_quasi_constant(self) -> bool:
        """Detects whether a single category almost entirely dominates the distribution.

        Quasi-constant features carry little information and may be candidates
        for removal in the pipeline.

        Returns:
            True if the most frequent category represents more than 95% of the data.
        """
        return self.most_frequent_ratio > 0.95


class CategoricalNominalFeature(CategoricalFeature):
    """Categorical feature with no intrinsic order between its categories.

    Represents variables such as country, color, or product type, where no
    defined ordering relationship exists between values.

    The encoding strategy is automatically selected based on cardinality,
    balancing dimensionality, performance, and leakage risk.

    Attributes:
        suggested_encoding: Recommended encoding derived automatically
            from cardinality and the presence of rare categories.

    Examples:
        >>> feature = CategoricalNominalFeature(
        ...     name="country", dtype="object", cardinality=5,
        ...     most_frequent_ratio=0.3
        ... )
        >>> feature.suggested_encoding
        <EncodingType.ONEHOT: 'onehot'>
    """

    feature_type: FeatureType = Field(default=FeatureType.CATEGORICAL_NOMINAL)

    @property
    def suggested_encoding(self) -> EncodingType:
        """Automatically selects the most appropriate encoding strategy.

        Selection criteria:
            - Very high cardinality (> 50): HASHING for memory efficiency.
            - High cardinality (> 20): TARGET or FREQUENCY to avoid dimensional explosion.
            - Rare categories present: FREQUENCY to handle unseen categories gracefully.
            - Low cardinality (<= 20): ONEHOT as the default option.

        Returns:
            The recommended EncodingType for this nominal feature.
        """
        if self.cardinality > 50:
            return EncodingType.HASHING
        if self.cardinality > 20:
            return EncodingType.TARGET if not self.has_rare_categories else EncodingType.FREQUENCY
        if self.has_rare_categories:
            return EncodingType.FREQUENCY
        return EncodingType.ONEHOT


class CategoricalOrdinalFeature(CategoricalFeature):
    """Categorical feature with a defined intrinsic order between its categories.

    Represents variables such as education level, t-shirt size (S/M/L/XL),
    or satisfaction rating (low/medium/high), where a meaningful ordering
    relationship exists between values.

    The order must be provided explicitly so the pipeline can build the correct
    mapping during OrdinalEncoder fitting.

    Attributes:
        category_order: Ordered list of categories from lowest to highest.
            Must contain exactly the same categories present in the dataset.

    Raises:
        ValueError: If category_order is provided as an empty list.

    Examples:
        >>> feature = CategoricalOrdinalFeature(
        ...     name="education_level", dtype="object", cardinality=4,
        ...     category_order=["primary", "secondary", "technical", "university"]
        ... )
        >>> feature.suggested_encoding
        <EncodingType.ORDINAL: 'ordinal'>
    """

    feature_type: FeatureType = Field(default=FeatureType.CATEGORICAL_ORDINAL)
    category_order: Optional[list[str]] = Field(
        default=None,
        description=(
            "Categories in ascending order. Required for OrdinalEncoder. "
            "Example: ['low', 'medium', 'high']."
        ),
    )

    @field_validator("category_order")
    @classmethod
    def validate_category_order_not_empty(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validates that the category order list is not empty when provided.

        Args:
            v: Ordered list of categories to validate.

        Returns:
            The validated list, or None if not provided.

        Raises:
            ValueError: If an empty list is provided.
        """
        if v is not None and len(v) == 0:
            raise ValueError("category_order cannot be an empty list.")
        return v

    @property
    def suggested_encoding(self) -> EncodingType:
        """Always returns ORDINAL as the encoding for features with a defined order.

        Returns:
            EncodingType.ORDINAL invariably.
        """
        return EncodingType.ORDINAL

    @property
    def has_defined_order(self) -> bool:
        """Checks whether the category order was explicitly specified.

        Returns:
            True if category_order was provided and contains at least one category.
        """
        return self.category_order is not None and len(self.category_order) > 0


# ---------------------------------------------------------------------------
# Datetime features
# ---------------------------------------------------------------------------


class DatetimeFeature(FeatureConfig):
    """Characterization of a temporal column (date, time, or timestamp).

    Date/time columns are rarely used directly in ML models. This feature class
    encapsulates the information needed to decide which temporal components to
    extract (year, month, day, hour, day of week, etc.) and whether to compute
    cyclic features or temporal deltas.

    Attributes:
        granularity: Temporal detail level detected in the column.
        min_date: Minimum date in the column as an ISO 8601 string (optional).
        max_date: Maximum date in the column as an ISO 8601 string (optional).
        has_timezone: True if values include timezone information.
        extract_components: Temporal components to extract in the pipeline.
            Defaults to year, month, and day of week.
        use_cyclic_encoding: True to encode cyclic components (month, day of week)
            as sin/cos pairs to preserve their circular nature.
        reference_date: Reference date for computing temporal deltas (elapsed days).
    """

    feature_type: FeatureType = Field(default=FeatureType.DATETIME)
    granularity: DatetimeGranularity = Field(
        default=DatetimeGranularity.DATE,
        description="Detected temporal granularity level.",
    )
    min_date: Optional[str] = Field(default=None, description="Minimum date (ISO 8601).")
    max_date: Optional[str] = Field(default=None, description="Maximum date (ISO 8601).")
    has_timezone: bool = Field(
        default=False, description="True if values include timezone information."
    )
    extract_components: list[str] = Field(
        default=["year", "month", "dayofweek"],
        description=(
            "Temporal components to extract. "
            "Options: year, month, day, hour, minute, dayofweek, quarter, weekofyear."
        ),
    )
    use_cyclic_encoding: bool = Field(
        default=True,
        description="True to encode month and day of week as sin/cos pairs.",
    )
    reference_date: Optional[str] = Field(
        default=None,
        description="Base date for computing elapsed days (temporal delta).",
    )

    @property
    def generates_delta_feature(self) -> bool:
        """Indicates whether the pipeline should generate an elapsed-days feature.

        Returns:
            True if a reference date was specified.
        """
        return self.reference_date is not None


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------


class TextFeature(FeatureConfig):
    """Characterization of a free-text or semi-structured column.

    Provides the metrics needed to select the most appropriate vectorization
    strategy based on text length, vocabulary size, and semantic nature.

    Attributes:
        avg_token_count: Average number of tokens (words) per record.
        vocabulary_size: Number of unique terms across the entire column.
        language: Detected language (ISO 639-1 code, e.g., 'es', 'en').
        is_semantic: True if the text has high semantic content (full sentences).
            False for short text or codes (e.g., textual categories, SKUs).
        max_features: Feature limit for TF-IDF or Count Vectorizer.

    Raises:
        ValueError: If avg_token_count or vocabulary_size are negative.
    """

    feature_type: FeatureType = Field(default=FeatureType.TEXT)
    avg_token_count: float = Field(
        default=1.0, ge=0.0, description="Average number of tokens per text record."
    )
    vocabulary_size: int = Field(
        default=0, ge=0, description="Number of unique terms in the corpus."
    )
    language: Optional[str] = Field(default=None, description="Detected language code (ISO 639-1).")
    is_semantic: bool = Field(
        default=False,
        description="True if the text is semantic (sentences). False for short text or codes.",
    )
    max_features: int = Field(default=5000, gt=0, description="Feature limit for vectorizers.")

    @property
    def suggested_vectorization(self) -> TextVectorizationType:
        """Automatically selects the most appropriate vectorization strategy.

        Selection criteria:
            - Semantic text: EMBEDDINGS to capture meaning.
            - Massive vocabulary (> 100k): HASH for efficiency.
            - Short or general text: TFIDF as a robust baseline.

        Returns:
            The most appropriate TextVectorizationType for this feature.
        """
        if self.is_semantic:
            return TextVectorizationType.EMBEDDINGS
        if self.vocabulary_size > 100_000:
            return TextVectorizationType.HASH
        return TextVectorizationType.TFIDF


# ---------------------------------------------------------------------------
# Boolean features
# ---------------------------------------------------------------------------


class BooleanFeature(FeatureConfig):
    """Characterization of a binary or boolean column.

    Handles columns with exactly two values (True/False, 0/1, Yes/No, etc.).
    These columns generally do not require transformation, but may need to be
    cast to integer (0/1) for compatibility with scikit-learn estimators.

    Attributes:
        true_ratio: Proportion of True/1 values in the column [0, 1].
        cast_to_int: True to convert bool to int in the pipeline.

    Raises:
        ValueError: If true_ratio is not in [0, 1].
    """

    feature_type: FeatureType = Field(default=FeatureType.BOOLEAN)
    true_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Proportion of True values [0, 1]."
    )
    cast_to_int: bool = Field(
        default=True, description="True to cast bool to int (0/1) in the pipeline."
    )

    @property
    def is_imbalanced(self) -> bool:
        """Detects whether the column has a severe imbalance between True and False.

        A ratio < 5% or > 95% indicates a near-constant feature that may
        contribute little predictive information.

        Returns:
            True if the proportion of True values is outside the [0.05, 0.95] range.
        """
        return self.true_ratio < 0.05 or self.true_ratio > 0.95


# ---------------------------------------------------------------------------
# Identifier / drop feature
# ---------------------------------------------------------------------------


class IdentifierFeature(FeatureConfig):
    """Identifier column that must be excluded from model training.

    Represents columns such as user IDs, primary keys, UUIDs, or any feature
    whose cardinality equals the number of rows and that provides no
    generalizable predictive signal.

    Attributes:
        cardinality: Number of unique values (generally equal to n_rows).
        is_primary_key: True if every value is unique across the dataset.
    """

    feature_type: FeatureType = Field(default=FeatureType.IDENTIFIER)
    drop: bool = Field(default=True, description="Always True: ID columns must be excluded.")
    cardinality: int = Field(..., gt=0, description="Number of unique values detected.")
    is_primary_key: bool = Field(
        default=True, description="True if each value is unique in the dataset."
    )
