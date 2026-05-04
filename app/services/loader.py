from typing import Tuple

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from app.exceptions.exceptions import FileIsEmptyException


class DatasetBundle(BaseModel):
    """Pydantic model to store train and test datasets.

    Attributes:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
    """

    model_config = {"arbitrary_types_allowed": True}

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class DataLoader:
    """Handles data loading, validation, and splitting operations."""

    @staticmethod
    def read_csv_file(file_path: str) -> pd.DataFrame:
        """Reads a CSV file into a pandas DataFrame.

        Args:
            file_path (str): The system path to the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The provided path {file_path} does not exist.")

    @staticmethod
    def validate_dataset(data: pd.DataFrame, target: str) -> None:
        """Performs basic validation on the dataset integrity.

        Args:
            data (pd.DataFrame): The dataset to validate.
            target (str): The name of the target column expected in the dataset.

        Raises:
            FileIsEmptyException: If the DataFrame contains no rows.
            ValueError: If the target column is missing from the DataFrame.
        """
        if data.empty:
            raise FileIsEmptyException("The file is empty.")

        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

    @staticmethod
    def compute_test_percentage(dataset_length: int) -> float:
        """Determines the test set ratio based on the size of the dataset.

        Args:
            dataset_length (int): Total number of records in the dataset.

        Returns:
            float: The calculated test size ratio (e.g., 0.1 or 0.2).
        """
        return 0.2 if dataset_length > 1000 else 0.1

    @staticmethod
    def split_dataset(
        data: pd.DataFrame,
        target: str,
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Splits the dataset into training and testing subsets.

        Args:
            data (pd.DataFrame): The complete dataset to be split.
            target (str): The name of the target column.

        Returns:
            Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
                A nested tuple containing: ((X_train, y_train), (X_test, y_test))
        """
        test_size = DataLoader.compute_test_percentage(len(data))

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=[target]),
            data[target],
            test_size=test_size,
            shuffle=True,
            random_state=42,
        )

        return (X_train, y_train), (X_test, y_test)

    def load_data(self, file_path: str, target: str) -> DatasetBundle:
        """Orchestrates the data ingestion pipeline.

        Args:
            file_path (str): Path to the CSV source file.
            target (str): The name of the column to be used as the label.

        Returns:
            DatasetBundle: The processed and split training and testing data.
        """
        data = self.read_csv_file(file_path)
        self.validate_dataset(data, target)
        (X_train, y_train), (X_test, y_test) = self.split_dataset(data, target)
        return DatasetBundle(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
