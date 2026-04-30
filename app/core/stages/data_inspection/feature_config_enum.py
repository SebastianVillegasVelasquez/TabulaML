from enum import Enum

from sklearn.preprocessing import MinMaxScaler


class TransformationType(str, Enum):
    """Transformation types applicable to numerical features.

    Attributes:
        NONE: No scale or distribution transformation required.
        LOG: Natural logarithm. Only valid when zero_ratio == 0 and skewness > 0.
        LOG1P: log(1 + x). Alternative to LOG when zeros are present but no negatives.
        YEO_JOHNSON: Yeo-Johnson transformation. Supports zeros and negative values.
        BOX_COX: Box-Cox transformation. Only valid for strictly positive values.
        SQRT: Square root. Mild alternative for moderate positive skewness.
        STANDARD: Standardization (z-score). For approximately normal distributions.
        MINMAX: Min-Max scaling. For bounded features without severe outliers.
        ROBUST: Robust scaling using median/IQR. For features with outliers.
    """

    NONE = "none"
    LOG = "log"
    LOG1P = "log1p"
    YEO_JOHNSON = "yeo_johnson"
    BOX_COX = "box_cox"
    SQRT = "sqrt"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class FeatureType(str, Enum):
    """Semantic type assigned to each column during data inspection.

    Inherits from str so the enum value remains JSON-serializable and
    compatible with Pydantic field validation without extra configuration.

    Attributes:
        NUMERICAL: Continuous or discrete numeric column.
        CATEGORICAL_NOMINAL: Unordered categorical column.
        CATEGORICAL_ORDINAL: Ordered categorical column.
        BOOLEAN: Binary column (True/False, 0/1, Yes/No).
        DATETIME: Date, time, or timestamp column.
        TEXT: Free-form text column.
        IDENTIFIER: ID-like column is excluded from training.
    """

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"


class NumericalSubtype(str, Enum):
    """Semantic subtype of a numerical column.

    Drives transformation selection independently of skewness, since count,
    cyclic, and binary-encoded columns have structural constraints that
    override distribution-based heuristics.

    Attributes:
        CONTINUOUS: Float or wide-range integer. Both order and magnitude matter.
        COUNT: Non-negative integer representing a frequency or occurrence.
            Typically Poisson-distributed; log1p is the natural transform.
        LOW_CARDINALITY_COUNT: Non-negative integer count with very few distinct
            values (<=6). The pipeline should consider one-hot encoding instead
            of scaling, since each value behaves more like a category.
        ORDINAL_ENCODED: Integer encoding of an ordered category (1=low, 3=high).
            Scaling should be avoided; treat as ordinal in the pipeline.
        CYCLIC: Integer with a known periodic range (hour 0-23, month 1-12).
            Requires sin/cos encoding to preserve circular distance.
        BINARY_ENCODED: Numeric 0/1 representing a boolean flag.
            No transformation needed.
    """

    CONTINUOUS = "continuous"
    COUNT = "count"
    LOW_CARDINALITY_COUNT = "low_cardinality_count"
    ORDINAL_ENCODED = "ordinal_encoded"
    CYCLIC = "cyclic"
    BINARY_ENCODED = "binary_encoded"


class EncodingType(str, Enum):
    """Encoding strategies for categorical features.

    Attributes:
        ONEHOT: One-Hot Encoding. Optimal for low-cardinality and linear/tree models.
        ORDINAL: Ordinal Encoding. For categories with an intrinsic defined order.
        TARGET: Target Encoding. For high cardinality it requires cross-validation to prevent leakage.
        FREQUENCY: Frequency Encoding. Replaces each category with its relative frequency.
        BINARY: Binary Encoding. Trade-off between One-Hot and Ordinal for medium cardinality.
        HASHING: Hashing Encoding. For very high cardinality with limited memory.
        LEAVE_ONE_OUT: Leave-One-Out Encoding. More robust variant of Target Encoding.
    """

    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"
    FREQUENCY = "frequency"
    BINARY = "binary"
    HASHING = "hashing"
    LEAVE_ONE_OUT = "leave_one_out"


class ImputationStrategy(str, Enum):
    """Missing value imputation strategies.

    Attributes:
        MEAN: Impute with the mean. Only for numerical features without severe outliers.
        MEDIAN: Impute with the median. Robust against outliers.
        MODE: Impute with the mode. For categorical or discrete numerical features.
        CONSTANT: Impute with a user-defined constant value.
        KNN: K-Nearest Neighbors Imputation. Costly but accurate.
        ITERATIVE: Iterative imputation (MICE). More precise, computationally intensive.
        INDICATOR: Adds a binary missingness indicator column before imputing.
    """

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"
    INDICATOR = "indicator"


class ScalerStrategy(str, Enum):
    """Scaling strategies for numerical features.

    Attributes:
        MIN_MAX_SCALER: Scales features to a range between 0 and 1.
        ROBUST_SCALER: Robust scaler that adapts to outliers.
        STANDARD_SCALER: Standard scaler that normalizes features to have zero mean and unit variance.
        MAX_ABS_SCALER: Scales features to have a maximum absolute value of 1.
    """

    MIN_MAX_SCALER = "min_max_scaler"
    ROBUST_SCALER = "robust"
    STANDARD_SCALER = "standard"
    MAX_ABS_SCALER = "max_abs_scaler"


class DatetimeGranularity(str, Enum):
    """Temporal granularity level detected in date/time features.

    Attributes:
        DATE: Date only (year, month, day).
        DATETIME: Full date and time.
        TIME: Time component only.
        YEAR_MONTH: Monthly granularity (e.g., invoices, reports).
        TIMESTAMP: Unix timestamp or similar.
    """

    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    YEAR_MONTH = "year_month"
    TIMESTAMP = "timestamp"


class TextVectorizationType(str, Enum):
    """Vectorization strategies for free-text features.

    Attributes:
        TFIDF: TF-IDF. Robust baseline for short to medium-length text.
        COUNT: Bag of Words with raw counts. Simple and fast.
        EMBEDDINGS: Semantic embeddings (e.g., sentence-transformers). For semantic text.
        HASH: HashingVectorizer. For massive vocabularies with limited memory.
    """

    TFIDF = "tfidf"
    COUNT = "count"
    EMBEDDINGS = "embeddings"
    HASH = "hash"
