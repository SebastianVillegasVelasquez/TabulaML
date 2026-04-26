from enum import Enum


class EvaluationType(Enum):
    """
    Enum for evaluation types.
    """

    DEFAULT = "default"
    THRESHOLD = "threshold"


#####################################################


class ExecutionStatus(Enum):
    """
    Represents the execution state of a pipeline stage.

    States:
    - PENDING: Stage is queued but not yet executing
    - RUNNING: Stage is currently executing
    - SUCCESS: Stage completed successfully
    - FAILED: Stage terminated with an error
    - SKIPPED: Stage was skipped due to failed preconditions
    """

    PENDING = "pending"  # Not yet started
    RUNNING = "running"  # In progress
    SUCCESS = "success"  # Completed successfully
    FAILED = "failed"  # Completed with error
    SKIPPED = "skipped"  # Preconditions aren't met


#####################################################


class ModelRetrieveType(Enum):
    BASELINE = "baseline"
    SELECTOR = "selector"
    PREDICTOR = "predictor"
    ENSEMBLER = "ensembler"


class ModelSpecType(Enum):
    LINEAR = "linear"
    NON_LINEAR = "non_linear"
    TREE = "tree"
    SVM = "svm"


class SelectorSpecType(Enum):
    STATISTICAL = "statistical"
    L1 = "l1"
    L2 = "l2"
    L1_L2 = "l1_l2"
    TREE_BASED = "tree_based"
    WRAPPER = "wrapper"
    FILTER = "filter"
    EXPLAINABLE = "explainable"
    RFE = "rfe"
    SHAP = "shap"


class EnsemblerSpecInfo(Enum):
    AVERAGE = "average"
    STACKING = "stacking"
    VOTING = "voting"


#####################################################


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


#####################################################


class Stages(Enum):
    """Enum representing the stages of the pipeline."""

    INITIALIZATION = "initialization"
    DATA_HANDLER = "data_handler"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    FINE_TUNING = "fine_tuning"
    MODEL_ENSEMBLE = "model_ensemble"
    MODEL_THRESHOLD_EXTRACTION = "threshold_selection"
    FINAL_EVALUATION = "final_evaluation"
    DEPLOYMENT = "deployment"
    EVALUATION = "evaluation"
