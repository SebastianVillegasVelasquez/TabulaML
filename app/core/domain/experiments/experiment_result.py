from typing import Dict, Any

from sklearn.pipeline import Pipeline
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    config: Dict[str, Any]
    coef: Dict[str, Any] | None = None