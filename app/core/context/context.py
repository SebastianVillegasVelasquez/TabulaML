from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import pandas as pd

from app.core.enums import ProblemType
from app.core.enums import Stages


@dataclass
class StageResult:
    name: Stages | str = None
    artifacts_path: str | None = None
    results: list[Any] | dict[str, Any] = field(default_factory=list)
    best_pipeline_path: str | Path = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    best_experiment: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectConfig:
    problem_type: ProblemType
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scoring: list[str] = field(default_factory=list)
    random_state: int = 42
    priority_metric: str | None = None
    priority_metric_normalized: str | None = None


@dataclass
class Context:
    """
    Stores the state of a full experimentation workflow.
    It is used to pass data between stages.
    """

    config: ProjectConfig | None = None

    current_stage: Stages | None = None

    stage_results: dict[Stages, StageResult] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def update_context(self, stage, stage_result: StageResult):
        if stage not in Stages.__members__.values():
            raise ValueError(f"Stage '{stage}' not registered.")
        self.current_stage = stage
        self.stage_results[stage] = stage_result
