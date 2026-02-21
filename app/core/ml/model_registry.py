import joblib
import json
from pathlib import Path

from app.core.domain.experiments.experiment import ExperimentResult


class ModelRegistry:
    """
    Stores trained pipelines and their metadata.
    """

    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def register(self, result: ExperimentResult, name: str) -> None:
        model_path = self.base_path / name
        model_path.mkdir(exist_ok=True)

        # Save pipeline
        joblib.dump(result.pipeline, model_path / "pipeline.joblib")

        # Save metadata
        metadata = {
            "metrics": result.metrics,
            "config": result.config
        }

        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, name: str):
        return joblib.load(self.base_path / name / "pipeline.joblib")
