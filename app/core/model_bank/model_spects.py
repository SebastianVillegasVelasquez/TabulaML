from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], BaseEstimator]
    __slots__ = ('name', 'factory')
@dataclass
class SelectorSpec:
    name: str
    factory: Callable[[], BaseEstimator]
    type: str
    __slots__ = ('name', 'factory', 'type')

@dataclass
class EnsemblerSpec:
    name: str
    factory: Callable[[], BaseEstimator]
    __slots__ = ('name', 'factory')

SELECTORS = [
    SelectorSpec(
        name="none",
        factory=lambda: None,
        type="baseline"
    ),
    SelectorSpec(
        name="selectkbest_f",
        factory=lambda: SelectKBest(score_func=f_classif, k=10),
        type="statistical"
    ),
    SelectorSpec(
        name="selectkbest_mi",
        factory=lambda: SelectKBest(score_func=mutual_info_classif, k=10),
        type="statistical"
    ),
]

