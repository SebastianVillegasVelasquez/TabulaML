from dataclasses import dataclass, field
from typing import Callable, Dict, Any

from sklearn.pipeline import Pipeline

from app.core.enums.stages import Stages
from app.core.ml.pipeline_builder import PipelineBuilder


@dataclass
class ExperimentDefinition:
    """
    Declarative experiment configuration.

    This object describes how to build an experiment
    but does not execute it.
    """

    name: str
    stage: str | Stages
    builder: Callable[..., PipelineBuilder] | Pipeline
    metadata: Dict[str, Any] | None = field(default_factory=dict)
