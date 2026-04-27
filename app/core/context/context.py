from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Tuple, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.model_selection import train_test_split

from app.core.enums import ProblemType
from app.core.enums import Stages
from app.core.metrics.metrics import DEFAULT_METRICS
from app.exceptions.exceptions import FileIsEmptyException


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

    def update_context(self, stage, stage_result: StageResult):
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
        """
        Factory method to create a Context instance with integrated data loading.

        Args:
            file_path: Path to the CSV file
            target_column: Name of the target column
            problem_type: Type of ML problem (CLASSIFICATION or REGRESSION)
            priority_metric: Priority metric name (optional)

        Returns:
            Context: Initialized context ready for pipeline execution
        """
        # Validate problem type
        if problem_type not in [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]:
            raise ValueError(f"Invalid problem type: {problem_type}")

        # Load and split data
        X, y = cls._load_data(file_path, target_column)

        # Create ProjectConfig
        config = ProjectConfig(
            problem_type=problem_type,
            scoring=DEFAULT_METRICS[problem_type],
            random_state=42,
            priority_metric=cls._get_priority_metric(problem_type, priority_metric),
            priority_metric_normalized=priority_metric,
            X_train=lambda: X[0],
            y_train=lambda: y[0],
            X_test=lambda: X[1],
            y_test=lambda: y[1],
        )

        # Create Metadata
        metadata = cls._get_metadata(X[0], target_column)

        # Create and return Context
        return cls(
            config=config,
            metadata=metadata,
        )

    @staticmethod
    def _load_data(
        file_path: str, target_column: str
    ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
        """Load and split dataset."""
        data = Context._read_csv_file(file_path)
        Context._validate_dataset(data, target_column)
        return Context._split_dataset(data, target_column)

    @staticmethod
    def _read_csv_file(file_path: str) -> pd.DataFrame:
        """Read CSV file with error handling."""
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The provided path {file_path} does not exist.")

    @staticmethod
    def _validate_dataset(data: pd.DataFrame, target: str) -> None:
        """Validate dataset integrity."""
        if data.empty:
            raise FileIsEmptyException("The file is empty.")
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

    @staticmethod
    def _split_dataset(
        data: pd.DataFrame, target: str
    ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
        """Split dataset into train and test sets."""
        test_size = 0.2 if len(data) > 1000 else 0.1

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[target]),
            data[target],
            test_size=test_size,
            shuffle=True,
            random_state=42,
        )

        return (X_train, X_test), (y_train, y_test)

    @staticmethod
    def _get_metadata(X: pd.DataFrame, target_column: str) -> Metadata:
        """Extract metadata from dataset."""
        return Metadata(
            columns=list(X.columns),
            columns_length=len(X.columns),
            target_column=target_column,
        )

    @staticmethod
    def _get_priority_metric(problem_type: ProblemType, priority_metric: Optional[str] = None) -> str:
        """Determine the priority metric for optimization."""
        if priority_metric is not None:
            return f"test_{priority_metric}"
        return (
            "test_f1" if problem_type == ProblemType.CLASSIFICATION else "test_neg_mean_squared_error"
        )

