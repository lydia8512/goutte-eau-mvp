import pandas as pd
import pytest

from model.train import validate_dataset


def test_validate_dataset_accepts_valid_data() -> None:
    dataframe = pd.DataFrame(
        {
            "temperature": [10.0, 12.0],
            "rain": [0, 1],
        }
    )

    validate_dataset(dataframe, "rain")


def test_validate_dataset_rejects_empty_data() -> None:
    dataframe = pd.DataFrame(columns=["rain"])

    with pytest.raises(ValueError, match="vide"):
        validate_dataset(dataframe, "rain")


def test_validate_dataset_rejects_missing_target() -> None:
    dataframe = pd.DataFrame(
        {
            "temperature": [10.0, 12.0],
        }
    )

    with pytest.raises(ValueError, match="absente"):
        validate_dataset(dataframe, "rain")


def test_validate_dataset_rejects_missing_target_values() -> None:
    dataframe = pd.DataFrame(
        {
            "rain": [0, None],
        }
    )

    with pytest.raises(ValueError, match="valeurs manquantes"):
        validate_dataset(dataframe, "rain")