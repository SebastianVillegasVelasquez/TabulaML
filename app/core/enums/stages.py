from enum import Enum


class Stages(Enum):
    """Enum representing the stages of the pipeline."""
    INITIALIZATION = "initialization"
    DATA_HANDLER = "data_handler"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    FINE_TUNING = "fine_tuning"
    MODEL_ENSEMBLE = "model_ensemble"
    MODEL_THRESHOLD_EXTRACTION = "threshold_selection"
    EVALUATION = "evaluation"

