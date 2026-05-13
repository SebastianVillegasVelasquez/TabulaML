from typing import Dict, Any

from sklearn.pipeline import Pipeline

from pydantic import BaseModel, ConfigDict, Field


class ExperimentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    pipeline: Pipeline | None = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    selected_features: list[Any] | None = Field(default_factory=list)
