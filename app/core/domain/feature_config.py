"""
This class is only data stored, the reason to not use a data type it is because
they easily access using a class
"""

class FeatureConfig:
    __slots__ = (
        "name",
        "dtype",
        "feature_type",
        "cardinality",
        "encoding",
        "is_categorical",
        "is_numerical",
        "skewness",
        "zero_ratio",
        "suggested_transformation"
    )

    def __init__(
            self,
            name: str,
            dtype: str,
            feature_type: str,
            cardinality: int | None = None,
            encoding: str | None = None,
            is_categorical: bool = False,
            is_numerical: bool = False,
            skewness: float | None = None,
            zero_ratio: float | None = None,
            suggested_transformation: str | None = None
    ):
        """
        Configuration object that stores metadata and preprocessing
        suggestions for a single feature.
        """
        self.name = name
        self.dtype = dtype
        self.feature_type = feature_type
        self.cardinality = cardinality
        self.encoding = encoding
        self.is_categorical = is_categorical
        self.is_numerical = is_numerical
        self.skewness = skewness
        self.zero_ratio = zero_ratio
        self.suggested_transformation = suggested_transformation

    def __str__(self) -> str:
        return (
            f"FeatureConfig("
            f"name={self.name}, "
            f"type={self.feature_type}, "
            f"dtype={self.dtype}, "
            f"cardinality={self.cardinality}, "
            f"encoding={self.encoding}, "
            f"skewness={self.skewness}, "
            f"transformation={self.suggested_transformation}"
            f")"
        )
