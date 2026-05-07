from typing import Dict, Any

from app.core.enums import EvaluationType
from app.core.enums import Stages
from app.core.stages.data_inspection.pipeline_builder import PipelineBuilder

from pydantic import BaseModel, Field


class ExperimentDefinition(BaseModel):
    """
    Declarative experiment configuration.

    This object describes how to build an experiment
    but does not execute it.
    """

    name: str
    stage: Stages
    pipeline_builder: PipelineBuilder
    evaluation_type: EvaluationType = EvaluationType.DEFAULT
    use_threshold: bool = False
    threshold: float = 0.5
    metadata: Dict[str, Any] | None = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
