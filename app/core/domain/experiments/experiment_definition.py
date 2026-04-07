from dataclasses import dataclass, field
from typing import Callable, Dict, Any

from sklearn.pipeline import Pipeline

from app.core.enums import EvaluationType
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
    pipeline_builder: PipelineBuilder
    evaluation_type: EvaluationType = EvaluationType.DEFAULT
    use_threshold: bool  = False
    threshold: float = 0.5
    metadata: Dict[str, Any] | None = field(default_factory=dict)
