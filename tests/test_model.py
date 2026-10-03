import joblib
import pandas as pd
import pytest

from model.train import (
    save_model,
    train_model,
    validate_dataset,
)


def build_training_dataframe() -> pd.DataFrame:
    """Construit un petit dataset temporel contenant les deux classes."""

    return pd.DataFrame(
        {
            "observed_at": pd.date_range(
                start="2026-01-01 00:00:00",
                periods=20,
                freq="h",
            ),
            "temperature": [
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
            ],
            "humidity": [
                40.0,
                70.0,
                42.0,
                72.0,
                44.0,
                74.0,
                46.0,
                76.0,
                48.0,
                78.0,
                50.0,
                80.0,
                52.0,
                82.0,
                54.0,
                84.0,
                56.0,
                86.0,
                58.0,
                88.0,
            ],
            "rain_next_hour": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )


def test_validate_dataset_accepts_valid_data() -> None:
    dataframe = pd.DataFrame(
        {
            "temperature": [10.0, 12.0],
            "rain": [0, 1],
        }
    )

    validate_dataset(
        dataframe,
        "rain",
    )


def test_validate_dataset_rejects_empty_dataset() -> None:
    dataframe = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="vide",
    ):
        validate_dataset(
            dataframe,
            "rain",
        )


def test_validate_dataset_rejects_missing_target() -> None:
    dataframe = pd.DataFrame(
        {
            "temperature": [10.0, 12.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="absente",
    ):
        validate_dataset(
            dataframe,
            "rain",
        )


def test_validate_dataset_rejects_missing_target_values() -> None:
    dataframe = pd.DataFrame(
        {
            "rain": [0, None],
        }
    )

    with pytest.raises(
        ValueError,
        match="valeurs manquantes",
    ):
        validate_dataset(
            dataframe,
            "rain",
        )


def test_train_model_returns_model_metrics_and_confusion_matrix() -> None:
    dataframe = build_training_dataframe()

    model, metrics, matrix = train_model(
        dataframe=dataframe,
        feature_columns=[
            "temperature",
            "humidity",
        ],
        target_column="rain_next_hour",
    )

    assert model is not None

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "average_precision",
        "roc_auc",
    }

    for value in metrics.values():
        assert 0.0 <= value <= 1.0

    assert len(matrix) == 2
    assert all(
        len(row) == 2
        for row in matrix
    )


def test_train_model_rejects_missing_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "temperature": [
                10.0,
                12.0,
            ],
            "rain": [
                0,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="absentes",
    ):
        train_model(
            dataframe=dataframe,
            feature_columns=[
                "temperature",
                "humidity",
            ],
            target_column="rain",
        )


def test_save_model_creates_joblib_file(
    tmp_path,
) -> None:
    dataframe = build_training_dataframe()

    feature_columns = [
        "temperature",
        "humidity",
    ]

    model, _, _ = train_model(
        dataframe=dataframe,
        feature_columns=feature_columns,
        target_column="rain_next_hour",
    )

    output_path = (
        tmp_path
        / "rain_model.joblib"
    )

    saved_path = save_model(
        model=model,
        feature_columns=feature_columns,
        output_path=output_path,
    )

    assert saved_path.exists()

    saved_content = joblib.load(
        saved_path
    )

    assert "model" in saved_content

    assert (
        saved_content["feature_columns"]
        == feature_columns
    )

    assert (
        saved_content["target"]
        == "rain_next_hour"
    )

    assert (
        saved_content["forecast_horizon"]
        == "1 hour"
    )