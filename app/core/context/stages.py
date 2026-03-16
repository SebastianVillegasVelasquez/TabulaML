from enum import Enum


class Stages(Enum):
    DATA_HANDLER = "data_handler"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    FINE_TUNING = "fine_tuning"

