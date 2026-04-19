from enum import Enum


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

class SelectorSpecInfo(Enum):
    STATISTICAL = "statistical"
    L1 = "l1"
    L2 = "l2"
    L1_L2 = "l1_l2"
    TREE_BASED = "tree_based"
    WRAPPER = "wrapper"
    FILTER = "filter"
    EXPLAINABLE = "explainable"
    SHAP = "shap"

class EnsemblerSpecInfo(Enum):
    AVERAGE = "average"
    STACKING = "stacking"
    VOTING = "voting"