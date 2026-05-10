import pandas as pd
import pytest

from app.exceptions.exceptions import FileIsEmptyException
from app.services.loader import DataLoader, DatasetBundle


@pytest.mark.parametrize(
    "dataset_size, expected",
    [
        (100, 0.1),
        (2000, 0.2),
    ],
)
def test_compute_test_percentage_parametrized(dataset_size, expected):
    assert DataLoader.compute_test_percentage(dataset_size) == expected


class TestValidateDataset:
    def test_raises_when_dataset_is_empty(self):
        df = pd.DataFrame()
        with pytest.raises(FileIsEmptyException):
            DataLoader.validate_dataset(df, "target")

    def test_raises_when_target_not_present(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(ValueError):
            DataLoader.validate_dataset(df, "target")

    def test_passes_when_valid_dataset(self):
        df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        DataLoader.validate_dataset(df, "target")


class TestSplitDataset:
    def test_split_preserves_total_length(self):
        df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})
        (X_train, y_train), (X_test, y_test) = DataLoader.split_dataset(df, "target")
        assert len(X_train) + len(X_test) == 20
        assert len(y_train) + len(y_test) == 20

    def test_target_not_in_features(self):
        df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})
        (X_train, _), _ = DataLoader.split_dataset(df, "target")
        assert "target" not in X_train.columns

    def test_split_returns_correct_types(self):
        df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})
        (X_train, y_train), (X_test, y_test) = DataLoader.split_dataset(df, "target")
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)


class TestReadCsvFile:
    def test_reads_valid_csv(self, tmp_path):
        df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        file_path = tmp_path / "data.csv"
        df.to_csv(file_path, index=False)
        loaded = DataLoader.read_csv_file(str(file_path))
        assert not loaded.empty
        assert list(loaded.columns) == ["feature", "target"]

    def test_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DataLoader.read_csv_file("non_existent.csv")

    def test_returns_dataframe(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        result = DataLoader.read_csv_file(str(file_path))
        assert isinstance(result, pd.DataFrame)


class TestLoadData:
    def test_load_data_full_flow(self, tmp_path):
        df = pd.DataFrame({"feature": range(50), "target": [0, 1] * 25})
        file_path = tmp_path / "data.csv"
        df.to_csv(file_path, index=False)

        loader = DataLoader()
        bundle = loader.load_data(str(file_path), "target")

        assert isinstance(bundle, DatasetBundle)
        assert len(bundle.X_train) + len(bundle.X_test) == 50
        assert "target" not in bundle.X_train.columns

    def test_load_data_returns_dataset_bundle(self, tmp_path):
        df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})
        file_path = tmp_path / "data.csv"
        df.to_csv(file_path, index=False)

        loader = DataLoader()
        bundle = loader.load_data(str(file_path), "target")

        assert isinstance(bundle.X_train, pd.DataFrame)
        assert isinstance(bundle.y_train, pd.Series)
        assert isinstance(bundle.X_test, pd.DataFrame)
        assert isinstance(bundle.y_test, pd.Series)
