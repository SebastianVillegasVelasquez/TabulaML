from dataclasses import dataclass, field
from typing import Any

from app.core.context.project_config import ProjectConfig
from app.core.context.stages import Stages


@dataclass
class StageResult:
    name: Stages | str = None
    artifacts_path: str | None = None
    results: list[Any] | dict[str, Any] = field(default_factory=list)
    best_pipeline_path: str | None = None


@dataclass
class RunContext:
    """
    Stores the state of a full experimentation workflow.
    """
    config: ProjectConfig

    current_stage: Stages | None = None

    stage_results: dict[Stages, StageResult] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def update_context(self, stage, stage_result: StageResult):
        if stage not in Stages.__members__.values():
            raise ValueError(f"Stage '{stage}' not registered.")
        self.current_stage = stage
        self.stage_results[stage] = stage_result
