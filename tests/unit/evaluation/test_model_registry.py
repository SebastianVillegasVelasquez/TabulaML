import pytest
import joblib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from app.core.stages.evaluation.model_registry import ModelRegistry
from app.core.domain.experiments.experiment_result import ExperimentResult
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class TestModelRegistry:

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create a temporary model registry for testing"""
        return ModelRegistry(base_path=str(tmp_path / "test_models"))

    @pytest.fixture
    def sample_result(self):
        """Create a sample ExperimentResult for testing"""
        pipeline = Pipeline([
            ("classifier", LogisticRegression())
        ])
        return ExperimentResult(
            name="test_experiment",
            pipeline=pipeline,
            metrics={"accuracy": 0.95, "precision": 0.92},
            config={"model": "LogisticRegression", "C": 1.0},
            selected_features=["feature1", "feature2", "feature3"]
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

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_creates_model_directory(self, mock_uuid, mock_datetime, temp_registry, sample_result):
        mock_dt = Mock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        mock_uuid_instance = Mock()
        mock_uuid_instance.hex = "abcdef123456"
        mock_uuid.return_value = mock_uuid_instance

        model_path = temp_registry.register(sample_result, "test_model")

        expected_dir = temp_registry.base_path / "test_model_20240101_120000_abcdef"
        assert Path(model_path) == expected_dir
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_saves_pipeline(self, mock_uuid, mock_datetime, temp_registry, sample_result):
        mock_dt = Mock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        mock_uuid_instance = Mock()
        mock_uuid_instance.hex = "abcdef123456"
        mock_uuid.return_value = mock_uuid_instance

        model_path = temp_registry.register(sample_result, "test_model")

        pipeline_file = Path(model_path) / "pipeline.joblib"
        assert pipeline_file.exists()

        loaded_pipeline = joblib.load(pipeline_file)
        assert isinstance(loaded_pipeline, Pipeline)
        assert "classifier" in loaded_pipeline.named_steps

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_saves_metadata(self, mock_uuid, mock_datetime, temp_registry, sample_result):
        mock_dt = Mock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        mock_uuid_instance = Mock()
        mock_uuid_instance.hex = "abcdef123456"
        mock_uuid.return_value = mock_uuid_instance

        model_path = temp_registry.register(sample_result, "test_model")

        metadata_file = Path(model_path) / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        assert metadata["name"] == "test_experiment"
        assert metadata["metrics"] == {"accuracy": 0.95, "precision": 0.92}
        assert metadata["config"] == {"model": "LogisticRegression", "C": 1.0}
        assert metadata["selected_features"] == ["feature1", "feature2", "feature3"]
        assert metadata["created_at"] == "20240101_120000"

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_returns_correct_path(self, mock_uuid, mock_datetime, temp_registry, sample_result):
        mock_dt = Mock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        mock_uuid_instance = Mock()
        mock_uuid_instance.hex = "abcdef123456"
        mock_uuid.return_value = mock_uuid_instance

        model_path = temp_registry.register(sample_result, "my_model")

        expected_path = str(temp_registry.base_path / "my_model_20240101_120000_abcdef")
        assert model_path == expected_path

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_multiple_models(self, mock_uuid, mock_datetime, temp_registry, sample_result):
        # First registration
        mock_dt1 = Mock()
        mock_dt1.strftime.return_value = "20240101_120000"

        mock_uuid_instance1 = Mock()
        mock_uuid_instance1.hex = "aaaaaa123456"

        # Second registration
        mock_dt2 = Mock()
        mock_dt2.strftime.return_value = "20240101_130000"

        mock_uuid_instance2 = Mock()
        mock_uuid_instance2.hex = "bbbbbb123456"

        mock_datetime.now.side_effect = [mock_dt1, mock_dt2]
        mock_uuid.side_effect = [mock_uuid_instance1, mock_uuid_instance2]

        path1 = temp_registry.register(sample_result, "model_v1")
        path2 = temp_registry.register(sample_result, "model_v2")

        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()

    def test_load_existing_model(self, temp_registry, sample_result):
        # First register a model
        model_path = temp_registry.register(sample_result, "test_model")
        model_name = Path(model_path).name

        # Then load it
        loaded_pipeline = temp_registry.load(model_name)

        assert isinstance(loaded_pipeline, Pipeline)
        assert "classifier" in loaded_pipeline.named_steps

    def test_load_nonexistent_model(self, temp_registry):
        with pytest.raises(FileNotFoundError):
            temp_registry.load("nonexistent_model")

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_with_empty_selected_features(self, mock_uuid, mock_datetime, temp_registry):
        mock_dt = Mock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        mock_uuid_instance = Mock()
        mock_uuid_instance.hex = "abcdef123456"
        mock_uuid.return_value = mock_uuid_instance

        result = ExperimentResult(
            name="test_exp",
            pipeline=Pipeline([("classifier", LogisticRegression())]),
            metrics={"accuracy": 0.85},
            config={},
            selected_features=[]
        )

        model_path = temp_registry.register(result, "test_model")

        metadata_file = Path(model_path) / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        assert metadata["selected_features"] == []

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_register_with_empty_metrics(self, mock_uuid, mock_datetime, temp_registry):
        mock_dt = Mock()
        mock_dt.strftime.return_value = "20240101_120000"
        mock_datetime.now.return_value = mock_dt

        mock_uuid_instance = Mock()
        mock_uuid_instance.hex = "abcdef123456"
        mock_uuid.return_value = mock_uuid_instance

        result = ExperimentResult(
            name="test_exp",
            pipeline=Pipeline([("classifier", LogisticRegression())]),
            metrics={},
            config={"param": "value"},
            selected_features=["f1"]
        )

        model_path = temp_registry.register(result, "test_model")

        metadata_file = Path(model_path) / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        assert metadata["metrics"] == {}

    @patch("datetime.datetime")
    @patch("uuid.uuid4")
    def test_unique_model_directories(self, mock_uuid, mock_datetime, temp_registry, sample_result):
        """Test that multiple registrations create unique directories"""
        # First registration
        mock_dt1 = Mock()
        mock_dt1.strftime.return_value = "20240101_120000"

        mock_uuid_instance1 = Mock()
        mock_uuid_instance1.hex = "aaaaaa123456"

        # Second registration
        mock_dt2 = Mock()
        mock_dt2.strftime.return_value = "20240101_120000"

        mock_uuid_instance2 = Mock()
        mock_uuid_instance2.hex = "bbbbbb123456"

        mock_datetime.now.side_effect = [mock_dt1, mock_dt2]
        mock_uuid.side_effect = [mock_uuid_instance1, mock_uuid_instance2]

        path1 = temp_registry.register(sample_result, "model")
        path2 = temp_registry.register(sample_result, "model")

        assert path1 != path2
        assert "aaaaaa" in path1
        assert "bbbbbb" in path2
