import pandas as pd
import pytest

from app.exceptions.exceptions import FileIsEmptyException
from app.services.loader import (
    read_csv_file,
    validate_dataset,
    compute_test_percentage,
    split_dataset,
    load_data,
)


@pytest.mark.parametrize(
    "dataset_size, expected",
    [
        (100, 0.1),
        (2000, 0.2),
    ],
)
def test_compute_test_percentage_parametrized(dataset_size, expected):
    assert compute_test_percentage(dataset_size) == expected


class TestValidateDataset:

    def test_raises_when_dataset_is_empty(self):
        df = pd.DataFrame()
        with pytest.raises(FileIsEmptyException):
            validate_dataset(df, "target")

    def test_raises_when_target_not_present(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(ValueError):
            validate_dataset(df, "target")

    def test_passes_when_valid_dataset(self):
        df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

        # Should not raise
        validate_dataset(df, "target")


class TestSplitDataset:

    def test_split_preserves_total_length(self):
        df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})

        (X_train, y_train), (X_test, y_test) = split_dataset(df, "target")

        assert len(X_train) + len(X_test) == 20
        assert len(y_train) + len(y_test) == 20

    def test_target_not_in_features(self):
        df = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})

        (X_train, _), _ = split_dataset(df, "target")

        assert "target" not in X_train.columns


class TestReadCsvFile:

    def test_reads_valid_csv(self, tmp_path):
        df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

        file_path = tmp_path / "data.csv"
        df.to_csv(file_path, index=False)

        loaded = read_csv_file(str(file_path))

        assert not loaded.empty
        assert list(loaded.columns) == ["feature", "target"]

    def test_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_csv_file("non_existent.csv")


class TestLoadData:

    def test_load_data_full_flow(self, tmp_path):
        df = pd.DataFrame({"feature": range(50), "target": [0, 1] * 25})

        file_path = tmp_path / "data.csv"
        df.to_csv(file_path, index=False)

        (X_train, y_train), (X_test, y_test) = load_data(str(file_path), "target")

        assert len(X_train) + len(X_test) == 50
        assert "target" not in X_train.columns
