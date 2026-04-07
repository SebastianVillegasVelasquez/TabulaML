from enum import Enum

class ModelRetrieveType(Enum):
    BASELINE = "baseline"
    SELECTOR = "selector"
    PREDICTOR = "predictor"
    ENSEMBLER = "ensembler"