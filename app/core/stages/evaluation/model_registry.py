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

        # Convert config to JSON-serializable format
        # serializable_config = self._make_serializable(result.config)
        serializable_metrics = self._make_serializable(result.metrics)

        if result.feature_mask is not None or result.selected_features is not None:
            serializable_selected_features = self._make_serializable(result.selected_features)
            metadata = {
                "name": result.name,
                "metrics": serializable_metrics,
                "selected_features": serializable_selected_features,
                "created_at": version
            }
        else :
            metadata = {
                "name": result.name,
                "metrics": serializable_metrics,
                "created_at": version
            }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return str(model_dir)

    def _make_serializable(self, obj):
        """Convert non-JSON-serializable objects to strings recursively."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, type):
            return f"{obj.__module__}.{obj.__name__}"
        elif hasattr(obj, "__class__") and not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj

    def load(self, name: str):
        return joblib.load(self.base_path / name / "pipeline.joblib")
