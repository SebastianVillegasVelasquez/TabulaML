from dataclasses import field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from app.core.enums import ProblemType
from app.core.enums import Stages
from .metrics import DEFAULT_METRICS
from app.services.loader import DatasetBundle


class StageResult(BaseModel):
    """Data model representing the output and artifacts of a specific pipeline stage.

    Attributes:
        name (Stages): The identifier of the execution stage.
        artifacts_path (str | None): Filesystem path where stage artifacts are stored.
        results (list[Any] | dict[str, Any]): Collection of metrics or objects
            produced during execution.
        best_pipeline_path (str | Path): Path to the serialized best-performing
            pipeline found in this stage.
        feature_importance (dict[str, float]): Mapping of feature names to their
            calculated importance scores.
        best_experiment (Any | None): Reference to the top-performing experiment
            metadata or object.
        metadata (dict[str, Any]): Additional unstructured information related
            to the stage execution.
    """

    name: Stages
    artifacts_path: str | None = None
    results: list[Any] | dict[str, Any] | None = None
    best_pipeline_path: str | Path = None
    best_experiment: Any | None = None
    metadata: list[Any] | dict[str, Any] | None = None


class ProjectConfig(BaseModel):
    """Configuration settings for the machine learning project and data.

    Attributes:
        model_config (ConfigDict): Pydantic configuration to allow arbitrary types.
        problem_type (ProblemType): The nature of the ML task (e.g., Classification).
        dataset (DatasetBundle): Container with train and test datasets.
        scoring (list[str]): List of metric names used for evaluation.
        random_state (int): Seed used for reproducibility. Defaults to 42.
        priority_metric (str | None): Primary metric used for model optimization.
        priority_metric_normalized (str | None): Human-readable name of the priority metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem_type: ProblemType
    dataset: DatasetBundle
    scoring: list[str] = field(default_factory=list)
    random_state: int = 42
    priority_metric: str | None = None
    priority_metric_normalized: str | None = None


class Metadata(BaseModel):
    """Schema information for the datasets used in the context.

    Attributes:
        columns (list[str]): List of feature column names.
        columns_length (int): Total count of feature columns.
        target_column (str): Name of the label/target column.
    """

    columns: list[str]
    columns_length: int
    target_column: str | None = None


class Context(BaseModel):
    """Orchestrator for the full experimentation workflow state.

    This class manages data loading, validation, and maintains the results
    of various execution stages, acting as a "Single Source of Truth"
    passed between pipeline components.

    Attributes:
        model_config (ConfigDict): Pydantic configuration to allow arbitrary types.
        config (ProjectConfig): The project-wide configuration settings.
        current_stage (Stages): The stage currently being executed.
        stage_results (dict[str, StageResult]): Registry of results indexed
            by stage name.
        metadata (dict[str, Any] | Metadata): Data schema and structural information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ProjectConfig
    current_stage: Stages = None
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] | Metadata = None

    @field_validator("config", mode="before")
    @classmethod
    def validate_config(cls, v: Any) -> ProjectConfig:
        """Validates that the provided configuration is a ProjectConfig instance.

        Args:
            v: The value to validate.

        Returns:
            ProjectConfig: The validated configuration instance.

        Raises:
            ValueError: If the input is not an instance of ProjectConfig.
        """
        if not isinstance(v, ProjectConfig):
            raise ValueError("config must be a ProjectConfig instance")
        return v

    def update_stage_context(self, stage: Any, stage_result: StageResult) -> None:
        """Updates the context with the results from a newly completed stage.

        Args:
            stage: The stage identifier to update.
            stage_result (StageResult): The result object containing stage outputs.

        Raises:
            ValueError: If the provided stage is not registered in the Stages enum.
        """
        if stage not in Stages.__members__.values():
            raise ValueError(f"Stage '{stage}' not registered.")
        self.current_stage = stage
        self.stage_results[stage] = stage_result

    @classmethod
    def create(
        cls,
        dataset: DatasetBundle,
        problem_type: ProblemType = ProblemType.CLASSIFICATION,
        priority_metric: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> "Context":
        """Factory method to initialize a new Context from a DatasetBundle.

        Args:
            dataset (DatasetBundle): Loaded and split dataset.
            problem_type (ProblemType): Classification or Regression.
                Defaults to CLASSIFICATION.
            priority_metric (Optional[str]): Metric to prioritize.
                Defaults to None.
            target_column (Optional[str]): Name of the target column for metadata.
                Defaults to None.

        Returns:
            Context: A fully initialized execution context.

        Raises:
            ValueError: If the problem_type is not supported.
        """
        if problem_type not in [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]:
            raise ValueError(f"Invalid problem model_based: {problem_type}")

        config = ProjectConfig(
            problem_type=problem_type,
            dataset=dataset,
            scoring=DEFAULT_METRICS[problem_type],
            random_state=42,
            priority_metric=cls._get_priority_metric(problem_type, priority_metric),
            priority_metric_normalized=priority_metric,
        )

        metadata = Metadata(
            columns=list(dataset.X_train.columns),
            columns_length=len(dataset.X_train.columns),
            target_column=target_column or None,
        )

        return cls(config=config, metadata=metadata)

    @staticmethod
    def _get_priority_metric(
        problem_type: ProblemType, priority_metric: Optional[str] = None
    ) -> str:
        """Internal helper to determine the priority metric string for optimization.

        Args:
            problem_type (ProblemType): The model_based of ML task.
            priority_metric (Optional[str]): User-defined priority metric.

        Returns:
            str: The formatted metric name (e.g., 'test_f1' or 'test_neg_mean_squared_error').
        """
        if priority_metric is not None:
            return f"test_{priority_metric}"
        return (
            "test_f1"
            if problem_type == ProblemType.CLASSIFICATION
            else "test_neg_mean_squared_error"
        )
