from dataclasses import dataclass
from typing import Callable, Any

from sklearn.base import BaseEstimator

from app.core.enums import (
    ModelSpecType,
    SelectorSpecInfo, EnsemblerSpecInfo)


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    spec_type: ModelSpecType
    type: ModelSpecType  # "linear", "non_linear", "tree", etc.
    __slots__ = ('name', 'factory', 'spec_type', 'type')


@dataclass
class SelectorSpec:
    name: str
    spec_type: ModelSpecType
    type: SelectorSpecInfo  # "statistical", "tree_based", etc.
    factory: Callable[[], Any]
    __slots__ = ('name', 'factory', 'spec_type', 'type')


@dataclass
class EnsemblerSpec:
    name: str
    spec_type: ModelSpecType
    type: EnsemblerSpecInfo
    factory: Callable[[], Any]
    __slots__ = ('name', 'factory', 'spec_type', 'type')
