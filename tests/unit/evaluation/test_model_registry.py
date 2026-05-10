import pytest
import joblib
import json
from pathlib import Path
from app.core.stages.evaluation.model_registry import ModelRegistry
from app.core.experiments import ExperimentResult
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class TestModelRegistry:
    @pytest.fixture
    def temp_registry(self, tmp_path):
        return ModelRegistry(base_path=str(tmp_path / "test_models"))

    @pytest.fixture
    def sample_result(self):
        return ExperimentResult(
            name="test_experiment",
            pipeline=Pipeline([("classifier", LogisticRegression())]),
            metrics={"accuracy": 0.95, "precision": 0.92},
            config={"model": "LogisticRegression", "C": 1.0},
            selected_features=["feature1", "feature2", "feature3"],
        )

    def test_init_creates_base_path(self, tmp_path):
        base_path = tmp_path / "new_models"
        assert not base_path.exists()
        registry = ModelRegistry(base_path=str(base_path))
        assert base_path.exists()
        assert base_path.is_dir()
        assert registry.base_path == base_path

    def test_init_with_existing_path(self, tmp_path):
        base_path = tmp_path / "existing_models"
        base_path.mkdir()
        registry = ModelRegistry(base_path=str(base_path))
        assert base_path.exists()
        assert registry.base_path == base_path

    def test_init_default_path(self):
        registry = ModelRegistry()
        assert registry.base_path == Path("models")

    def test_register_creates_model_directory(self, temp_registry, sample_result):
        model_path = temp_registry.register(sample_result, "test_model")
        assert Path(model_path).exists()
        assert Path(model_path).is_dir()
        assert "test_model" in model_path

    def test_register_saves_pipeline(self, temp_registry, sample_result):
        model_path = temp_registry.register(sample_result, "test_model")
        pipeline_file = Path(model_path) / "pipeline.joblib"
        assert pipeline_file.exists()
        loaded_pipeline = joblib.load(pipeline_file)
        assert isinstance(loaded_pipeline, Pipeline)
        assert "classifier" in loaded_pipeline.named_steps

    def test_register_saves_metadata(self, temp_registry, sample_result):
        model_path = temp_registry.register(sample_result, "test_model")
        metadata_file = Path(model_path) / "metadata.json"
        assert metadata_file.exists()
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        assert metadata["name"] == "test_experiment"
        assert metadata["metrics"] == {"accuracy": 0.95, "precision": 0.92}
        assert "config" not in metadata
        assert metadata["selected_features"] == ["feature1", "feature2", "feature3"]
        assert "created_at" in metadata

    def test_register_returns_correct_path(self, temp_registry, sample_result):
        model_path = temp_registry.register(sample_result, "my_model")
        assert "my_model" in model_path
        assert Path(model_path).exists()

    def test_register_multiple_models(self, temp_registry, sample_result):
        path1 = temp_registry.register(sample_result, "model_v1")
        path2 = temp_registry.register(sample_result, "model_v2")
        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()

    def test_load_existing_model(self, temp_registry, sample_result):
        model_path = temp_registry.register(sample_result, "test_model")
        model_name = Path(model_path).name
        loaded_pipeline = temp_registry.load(model_name)
        assert isinstance(loaded_pipeline, Pipeline)
        assert "classifier" in loaded_pipeline.named_steps

    def test_load_nonexistent_model(self, temp_registry):
        with pytest.raises(FileNotFoundError):
            temp_registry.load("nonexistent_model")

    def test_register_with_empty_selected_features(self, temp_registry):
        result = ExperimentResult(
            name="test_exp",
            pipeline=Pipeline([("classifier", LogisticRegression())]),
            metrics={"accuracy": 0.85},
            config={},
            selected_features=[],
        )
        model_path = temp_registry.register(result, "test_model")
        metadata_file = Path(model_path) / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        assert metadata["selected_features"] == []

    def test_register_with_empty_metrics(self, temp_registry):
        result = ExperimentResult(
            name="test_exp",
            pipeline=Pipeline([("classifier", LogisticRegression())]),
            metrics={},
            config={"param": "value"},
            selected_features=["f1"],
        )
        model_path = temp_registry.register(result, "test_model")
        metadata_file = Path(model_path) / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        assert metadata["metrics"] == {}

    def test_unique_model_directories(self, temp_registry, sample_result):
        path1 = temp_registry.register(sample_result, "model")
        path2 = temp_registry.register(sample_result, "model")
        assert path1 != path2
