from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class ExperimentResult:
    name: str | None = None
    pipeline: Pipeline = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] | None = field(default=list)
    feature_mask: np.ndarray | None = None
