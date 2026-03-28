from enum import Enum


class TunerStrategy(Enum):
    OPTUNA = "optuna"
    GRID_SEARCH = "grid_search"