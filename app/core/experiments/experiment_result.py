from typing import Dict, Any, DefaultDict

import numpy as np
from sklearn.pipeline import Pipeline

from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    name: str | None = None
    pipeline: Pipeline = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    selected_features: list[str] | None = Field(default=list)

    model_config = DefaultDict(arbitrary_types_allowed=True)
