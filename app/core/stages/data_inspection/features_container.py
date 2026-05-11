"""
Automatic preprocessing pipeline builder for the AutoML system.

This module converts the typed feature metadata produced by DataInspectionStage
into a fitted-ready sklearn ColumnTransformer. The builder never fits or
transforms data — it only assembles the graph of transformers based on the
decisions already encoded in each feature's computed properties.

Architecture overview:

    FeatureContainer (grouped by FeatureType)
        └── PreprocessingBuilder
                ├── _group_features_by_type()    → list[FeatureContainer]
                ├── _build_numerical_pipeline()  → transformers for numerical cols
                ├── _build_categorical_pipeline()→ transformers for categorical cols
                ├── _build_boolean_pipeline()    → transformer  for boolean cols
                └── build() → ColumnTransformer (with set_output="pandas")

Column name preservation:
    Every sub-pipeline calls .set_output(transform="pandas") so the final
    ColumnTransformer returns a DataFrame with human-readable column names.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from app.utils.logger import logger
from .feature_config import (
    BooleanFeature,
    CategoricalNominalFeature,
    CategoricalOrdinalFeature,
    EncodingType,
    ImputationStrategy,
    NumericalFeature,
    NumericalSubtype,
    ScalerStrategy,
    TransformationType,
)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class FeatureContainer(BaseModel):
    """Groups feature configs that share the same semantic FeatureType.

    Used as an intermediate grouping structure between DataInspectionStage
    and the pipeline builder. Each container holds all features of one model_based
    (e.g., all NUMERICAL features, all CATEGORICAL_NOMINAL features).

    Attributes:
        feature_type: The shared FeatureType of every feature in this container.
        features: List of typed FeatureConfig subclass instances.
    """

    model_config = {"arbitrary_types_allowed": True}

    feature_type: Enum
    features: list[Any]


# ---------------------------------------------------------------------------
# Imputer factory
# ---------------------------------------------------------------------------


def _build_imputer(strategy: ImputationStrategy) -> SimpleImputer:
    """Instantiates a SimpleImputer from an ImputationStrategy enum value.

    Centralizes the mapping between the domain enum and sklearn's string-based
    strategy parameter so it is defined in exactly one place.

    Args:
        strategy: The ImputationStrategy selected by the feature's property.

    Returns:
        A configured but unfitted SimpleImputer instance.

    Raises:
        ValueError: If the strategy cannot be mapped to a SimpleImputer.
            CONSTANT, KNN, ITERATIVE, and INDICATOR require separate handling.
    """
    mapping: dict[ImputationStrategy, str] = {
        ImputationStrategy.MEAN: "mean",
        ImputationStrategy.MEDIAN: "median",
        ImputationStrategy.MODE: "most_frequent",
    }
    sklearn_strategy = mapping.get(strategy)
    if sklearn_strategy is None:
        raise ValueError(
            f"ImputationStrategy.{strategy.name} cannot be mapped to a SimpleImputer. "
            f"CONSTANT, KNN, ITERATIVE, and INDICATOR require separate handling."
        )
    return SimpleImputer(strategy=sklearn_strategy)


# ---------------------------------------------------------------------------
# Scaler factory
# ---------------------------------------------------------------------------


def _build_scaler(strategy: ScalerStrategy) -> Any:
    """Instantiates a sklearn scaler from a ScalerStrategy enum value.

    Args:
        strategy: The ScalerStrategy selected by the feature's property.

    Returns:
        A configured but unfitted sklearn scaler instance.

    Raises:
        ValueError: If the strategy has no registered scaler.
    """
    from sklearn.preprocessing import MaxAbsScaler

    mapping: dict[ScalerStrategy, Any] = {
        ScalerStrategy.STANDARD_SCALER: StandardScaler(),
        ScalerStrategy.ROBUST_SCALER: RobustScaler(),
        ScalerStrategy.MIN_MAX_SCALER: MinMaxScaler(),
        ScalerStrategy.MAX_ABS_SCALER: MaxAbsScaler(),
    }
    scaler = mapping.get(strategy)
    if scaler is None:
        raise ValueError(
            f"ScalerStrategy.{strategy.name} has no registered sklearn scaler."
        )
    return scaler


# ---------------------------------------------------------------------------
# Transformation factory
# ---------------------------------------------------------------------------


def _build_transform_step(transformation: TransformationType) -> tuple[str, Any] | None:
    """Returns a named (step_name, transformer) tuple for a TransformationType.

    All FunctionTransformer instances use feature_names_out='one-to-one'
    because mathematical transformations (log, sqrt, etc.) are element-wise
    and always return the same number of columns with the same names.
    Without this, Pipeline.get_feature_names_out() raises AttributeError
    since FunctionTransformer does not implement it by default.

    Args:
        transformation: The TransformationType selected by the feature's property.

    Returns:
        A (name, transformer) tuple ready for Pipeline, or None if NONE.
    """
    if transformation == TransformationType.NONE:
        return None

    if transformation == TransformationType.LOG:
        return (
            "log",
            FunctionTransformer(np.log, validate=True, feature_names_out="one-to-one"),
        )

    if transformation == TransformationType.LOG1P:
        return (
            "log1p",
            FunctionTransformer(
                np.log1p, validate=True, feature_names_out="one-to-one"
            ),
        )

    if transformation == TransformationType.SQRT:
        return (
            "sqrt",
            FunctionTransformer(np.sqrt, validate=True, feature_names_out="one-to-one"),
        )

    if transformation == TransformationType.YEO_JOHNSON:
        return ("yeo_johnson", PowerTransformer(method="yeo-johnson"))

    if transformation == TransformationType.BOX_COX:
        return ("box_cox", PowerTransformer(method="box-cox"))

    if transformation == TransformationType.STANDARD:
        return ("standard", StandardScaler())

    if transformation == TransformationType.ROBUST:
        return ("robust", RobustScaler())

    if transformation == TransformationType.MINMAX:
        return ("minmax", MinMaxScaler())

    logger.warning(
        "TransformationType.%s has no registered transformer. Skipping.",
        transformation.name,
    )
    return None


# ---------------------------------------------------------------------------
# Numerical pipeline builder
# ---------------------------------------------------------------------------


def _build_numerical_pipeline(features: list[NumericalFeature]) -> list[tuple]:
    """Builds ColumnTransformer-ready tuples for a list of NumericalFeature instances.

    Splits features into two tracks:
        Scale track: features with a suggested_scaler (CONTINUOUS, COUNT).
            Grouped by (transformation, scaler, imputer) signature so columns
            sharing identical steps use one Pipeline instance.
        Encode track: features without a suggested_scaler (LOW_CARDINALITY_COUNT,
            CYCLIC, ORDINAL_ENCODED, BINARY_ENCODED). Routed to encoding pipelines.

    Args:
        features: All NumericalFeature instances from the feature registry.

    Returns:
        List of (name, Pipeline, column_names) tuples for ColumnTransformer.
    """
    to_encode: list[NumericalFeature] = []
    to_scale: dict[tuple, list[str]] = defaultdict(list)

    for f in features:
        if f.suggested_scaler is None:
            to_encode.append(f)
        else:
            signature = (
                f.suggested_transformation,
                f.suggested_scaler,
                f.suggested_imputer,
            )
            to_scale[signature].append(f.name)

    transformers: list[tuple] = []

    # --- Scale track: one Pipeline per unique (transform, scaler, imputer) signature ---
    for i, (signature, cols) in enumerate(to_scale.items()):
        transformation, scaler_strategy, imputer_strategy = signature

        steps: list[tuple[str, Any]] = [("imputer", _build_imputer(imputer_strategy))]

        transform_step = _build_transform_step(transformation)
        if transform_step is not None:
            steps.append(transform_step)

        steps.append(("scaler", _build_scaler(scaler_strategy)))

        pipeline = Pipeline(steps).set_output(transform="pandas")
        name = f"num_scaled_{i}"
        transformers.append((name, pipeline, cols))
        logger.debug(
            "Numerical scale pipeline '%s': cols=%s | transform=%s scaler=%s imputer=%s",
            name,
            cols,
            transformation.value,
            scaler_strategy.value,
            imputer_strategy.value,
        )

    # --- Encode track: structural subtypes that should not be scaled ---
    if to_encode:
        transformers.extend(_build_numerical_encoding_pipelines(to_encode))

    return transformers


def _build_numerical_encoding_pipelines(
    features: list[NumericalFeature],
) -> list[tuple]:
    """Routes structural numerical subtypes to appropriate encoding pipelines.

    Routing logic per subtype:
        BINARY_ENCODED        → passthrough (already 0/1, no processing needed)
        CYCLIC                → passthrough (sin/cos encoding not yet implemented)
        ORDINAL_ENCODED       → OrdinalEncoder with median imputation
        LOW_CARDINALITY_COUNT → OneHotEncoder with most_frequent imputation

    Args:
        features: NumericalFeature instances whose suggested_scaler is None.

    Returns:
        List of (name, Pipeline, column_names) tuples for ColumnTransformer.
    """
    passthrough_cols: list[str] = []
    ordinal_cols: list[str] = []
    ohe_cols: list[str] = []

    for f in features:
        if f.subtype in (NumericalSubtype.BINARY_ENCODED, NumericalSubtype.CYCLIC):
            passthrough_cols.append(f.name)
        elif f.subtype == NumericalSubtype.ORDINAL_ENCODED:
            ordinal_cols.append(f.name)
        else:
            # LOW_CARDINALITY_COUNT → one-hot encode
            ohe_cols.append(f.name)

    transformers: list[tuple] = []

    if passthrough_cols:
        transformers.append(("num_passthrough", "passthrough", passthrough_cols))
        logger.debug("Numerical passthrough cols: %s", passthrough_cols)

    if ordinal_cols:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        ).set_output(transform="pandas")
        transformers.append(("num_ordinal", pipeline, ordinal_cols))
        logger.debug("Numerical ordinal-encoded cols: %s", ordinal_cols)

    if ohe_cols:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        ).set_output(transform="pandas")
        transformers.append(("num_ohe", pipeline, ohe_cols))
        logger.debug("Numerical OHE cols (low cardinality count): %s", ohe_cols)

    return transformers


# ---------------------------------------------------------------------------
# Categorical pipeline builder
# ---------------------------------------------------------------------------


def _build_categorical_pipeline(
    nominal: list[CategoricalNominalFeature],
    ordinal: list[CategoricalOrdinalFeature],
) -> list[tuple]:
    """Builds ColumnTransformer-ready tuples for categorical features.

    Nominal features are grouped by their (encoding, high_missing) signature
    so that columns with identical steps share one Pipeline. Ordinal features
    are grouped separately and pass their category_order lists to OrdinalEncoder.

    Args:
        nominal: All CategoricalNominalFeature instances.
        ordinal: All CategoricalOrdinalFeature instances.

    Returns:
        List of (name, Pipeline, column_names) tuples for ColumnTransformer.
    """
    transformers: list[tuple] = []

    # --- Nominal ---
    nominal_groups: dict[tuple, list[CategoricalNominalFeature]] = defaultdict(list)
    for f in nominal:
        signature = (f.suggested_encoding, f.is_high_missing)
        nominal_groups[signature].append(f)

    for i, (signature, group) in enumerate(nominal_groups.items()):
        encoding, high_missing = signature
        cols = [f.name for f in group]

        if high_missing:
            logger.warning(
                "Nominal group %s has high missingness (>40%%). "
                "Consider adding a MissingIndicator upstream.",
                cols,
            )

        steps = _build_nominal_steps(encoding)
        pipeline = Pipeline(steps).set_output(transform="pandas")
        name = f"cat_nominal_{i}_{encoding.value}"
        transformers.append((name, pipeline, cols))
        logger.debug(
            "Nominal pipeline '%s': cols=%s encoding=%s", name, cols, encoding.value
        )

    # --- Ordinal ---
    if ordinal:
        cols = [f.name for f in ordinal]
        categories_param = [
            f.category_order if f.has_defined_order else "auto" for f in ordinal
        ]
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=categories_param,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        ).set_output(transform="pandas")
        transformers.append(("cat_ordinal", pipeline, cols))
        logger.debug("Ordinal pipeline: cols=%s", cols)

    return transformers


def _build_nominal_steps(encoding: EncodingType) -> list[tuple[str, Any]]:
    """Assembles the imputer + encoder step list for a nominal categorical Pipeline.

    The imputer always uses most_frequent because the mode is the only
    statistically valid fill value for unordered categories.

    Unimplemented strategies (TARGET, FREQUENCY, HASHING, LEAVE_ONE_OUT)
    fall back to OneHotEncoder with a logged warning.

    Args:
        encoding: The EncodingType suggested by the feature's suggested_encoding
            property.

    Returns:
        Ordered list of (step_name, transformer) tuples for Pipeline().
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]

    if encoding == EncodingType.ONEHOT:
        steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        )

    elif encoding in (EncodingType.ORDINAL, EncodingType.BINARY):
        steps.append(
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        )

    else:
        # TARGET, FREQUENCY, HASHING, LEAVE_ONE_OUT — not yet implemented
        logger.warning(
            "EncodingType.%s is not yet implemented. Falling back to OneHotEncoder.",
            encoding.name,
        )
        steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        )

    return steps


# ---------------------------------------------------------------------------
# Boolean pipeline builder
# ---------------------------------------------------------------------------


def _build_boolean_pipeline(features: list[BooleanFeature]) -> list[tuple]:
    """Builds encode-and-cast pipelines for boolean features.

    Splits features into two sub-groups based on dtype, because the
    safe cast strategy differs:

        Numeric/bool dtypes (int64, float64, bool):
            SimpleImputer → FunctionTransformer(cast to int)
            These already contain 0/1 or True/False values that Python
            can cast to int directly.

        String dtypes (object, str):
            SimpleImputer → OrdinalEncoder
            OrdinalEncoder maps each unique string to 0 or 1 safely,
            avoiding the ValueError that int('male') would raise.
            No extra cast step needed — OrdinalEncoder already outputs float.

    Args:
        features: All BooleanFeature instances from the feature registry.

    Returns:
        Up to two (name, Pipeline, column_names) tuples — one per sub-group —
        or an empty list if no boolean features are provided.
    """
    if not features:
        return []

    numeric_cols: list[str] = []
    string_cols: list[str] = []

    for f in features:
        if f.is_imbalanced:
            logger.warning(
                "Boolean feature '%s' is severely imbalanced (true_ratio=%.3f). "
                "It may contribute little predictive signal.",
                f.name,
                f.true_ratio,
            )
        # Route by dtype — object/str columns cannot be cast to int directly
        if any(t in f.dtype for t in ("int", "float", "bool")):
            numeric_cols.append(f.name)
        else:
            string_cols.append(f.name)

    transformers: list[tuple] = []

    if numeric_cols:
        # FunctionTransformer wraps a plain Python function into a sklearn
        # transformer. validate=False lets non-numeric arrays pass through
        # without triggering sklearn's input validation checks.
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "cast_int",
                    FunctionTransformer(
                        lambda X: X.astype(int),
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
            ]
        ).set_output(transform="pandas")
        transformers.append(("boolean_numeric", pipeline, numeric_cols))
        logger.debug("Boolean numeric pipeline cols: %s", numeric_cols)

    if string_cols:
        # OrdinalEncoder is the correct tool here — it learns the mapping
        # {'female': 0, 'male': 1} from the training data and applies it
        # consistently at transform time, handling unseen values gracefully.
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        ).set_output(transform="pandas")
        transformers.append(("boolean_string", pipeline, string_cols))
        logger.debug("Boolean string pipeline cols: %s", string_cols)

    return transformers
