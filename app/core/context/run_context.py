from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.context.project_config import ProjectConfig
from app.core.context.stages import Stages


@dataclass
class StageResult:
    name: Stages | str = None
    artifacts_path: str | None = None
    results: list[Any] | dict[str, Any] = field(default_factory=list)
    best_pipeline_path: str | Path = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    best_experiment: Any | None = None


@dataclass
class RunContext:
    """
    Stores the state of a full experimentation workflow.
    """
    config: ProjectConfig | None = None

    current_stage: Stages | None = None

    stage_results: dict[Stages, StageResult] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def update_context(self, stage, stage_result: StageResult):
        if stage not in Stages.__members__.values():
            raise ValueError(f"Stage '{stage}' not registered.")
        self.current_stage = stage
        self.stage_results[stage] = stage_result
