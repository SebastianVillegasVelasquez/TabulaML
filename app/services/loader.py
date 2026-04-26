from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from app.exceptions.exceptions import FileIsEmptyException


# TODO: Managed 2 different data set if the user provided them
def read_csv_file(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The provided path {file_path} does not exist.")


def validate_dataset(data: pd.DataFrame, target: str) -> None:
    if data.empty:
        raise FileIsEmptyException("The file is empty.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")


def compute_test_percentage(dataset_length: int) -> float:
    return 0.2 if dataset_length > 1000 else 0.1


def split_dataset(
    data: pd.DataFrame,
    target: str,
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    test_size = compute_test_percentage(len(data))

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=[target]),
        data[target],
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )

    return (X_train, y_train), (X_test, y_test)


def load_data(file_path: str, target: str):
    data = read_csv_file(file_path)
    validate_dataset(data, target)
    return split_dataset(data, target)
