from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from app.core.enums import ProblemType
from app.core.enums import Stages
from .metrics import DEFAULT_METRICS
from app.services.loader import load_data


class StageResult(BaseModel):
    name: Stages
    artifacts_path: str | None = None
    results: list[Any] | dict[str, Any] = field(default_factory=list)
    best_pipeline_path: str | Path = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    best_experiment: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem_type: ProblemType
    X_train: Callable[[], pd.DataFrame]
    y_train: Callable[[], pd.Series]
    X_test: Callable[[], pd.DataFrame]
    y_test: Callable[[], pd.Series]
    scoring: list[str] = field(default_factory=list)
    random_state: int = 42
    priority_metric: str | None = None
    priority_metric_normalized: str | None = None


class Metadata(BaseModel):
    columns: list[str]
    columns_length: int
    target_column: str


class Context(BaseModel):
    """
    Stores the state of a full experimentation workflow.
    It handles its own creation, data loading, and validation.
    It is used to pass data between stages.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ProjectConfig
    current_stage: Stages = None
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] | Metadata = None

    @field_validator('config', mode='before')
    @classmethod
    def validate_config(cls, v):
        if not isinstance(v, ProjectConfig):
            raise ValueError("config must be a ProjectConfig instance")
        return v

    def update_stage_context(self, stage, stage_result: StageResult):
        if stage not in Stages.__members__.values():
            raise ValueError(f"Stage '{stage}' not registered.")
        self.current_stage = stage
        self.stage_results[stage] = stage_result

    @classmethod
    def create(
        cls,
        file_path: str,
        target_column: str,
        problem_type: ProblemType = ProblemType.CLASSIFICATION,
        priority_metric: Optional[str] = None,
    ) -> "Context":
        if problem_type not in [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]:
            raise ValueError(f"Invalid problem type: {problem_type}")

        (X_train, y_train), (X_test, y_test) = load_data(file_path, target_column)

        config = ProjectConfig(
            problem_type=problem_type,
            scoring=DEFAULT_METRICS[problem_type],
            random_state=42,
            priority_metric=cls._get_priority_metric(problem_type, priority_metric),
            priority_metric_normalized=priority_metric,
            X_train=lambda: X_train,
            y_train=lambda: y_train,
            X_test=lambda: X_test,
            y_test=lambda: y_test,
        )

        metadata = Metadata(
            columns=list(X_train.columns),
            columns_length=len(X_train.columns),
            target_column=target_column,
        )

        return cls(config=config, metadata=metadata)

    @staticmethod
    def _get_priority_metric(problem_type: ProblemType, priority_metric: Optional[str] = None) -> str:
        """Determine the priority metric for optimization."""
        if priority_metric is not None:
            return f"test_{priority_metric}"
        return (
            "test_f1" if problem_type == ProblemType.CLASSIFICATION else "test_neg_mean_squared_error"
        )