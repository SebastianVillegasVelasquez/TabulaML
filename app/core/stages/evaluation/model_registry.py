import joblib
import json
from pathlib import Path

from app.core.domain.experiments.experiment_result import ExperimentResult


class ModelRegistry:
    """
    Stores trained pipelines and their metadata.
    """

    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def register(self, result: ExperimentResult, name: str) -> str:
        import uuid
        from datetime import datetime, timezone

        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]

        model_dir = self.base_path / f"{name}_{version}_{unique_id}"
        model_dir.mkdir(exist_ok=True)

        joblib.dump(result.pipeline, model_dir / "pipeline.joblib")

        metadata = {
            "name": result.name,
            "metrics": result.metrics,
            "config": result.config,
            "selected_features": result.selected_features,
            "created_at": version
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return str(model_dir)

    def load(self, name: str):
        return joblib.load(self.base_path / name / "pipeline.joblib")
