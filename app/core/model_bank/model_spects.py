from typing import Callable, Any

from pydantic import BaseModel

from app.core.enums import ModelSpecType, SelectorSpecType, EnsemblerSpecInfo


class ModelSpec(BaseModel):
    name: str
    factory: Callable[[], Any]
    spec_type: ModelSpecType
    type: ModelSpecType  # "linear", "non_linear", "tree", etc.
    __slots__ = ("name", "factory", "spec_type", "type")


class SelectorSpec(BaseModel):
    name: str
    spec_type: ModelSpecType
    type: SelectorSpecType  # "statistical", "tree_based", etc.
    factory: Callable[[], Any]
    __slots__ = ("name", "factory", "spec_type", "type")


class EnsemblerSpec(BaseModel):
    name: str
    spec_type: ModelSpecType
    type: EnsemblerSpecInfo
    factory: Callable[[], Any]
    __slots__ = ("name", "factory", "spec_type", "type")
