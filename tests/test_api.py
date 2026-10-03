import numpy as np
from fastapi.testclient import TestClient

from api import main as api_main


client = TestClient(api_main.app)


class FakeModel:
    """Modèle factice utilisé uniquement pour les tests API."""

    def predict_proba(self, dataframe):
        count = len(dataframe)

        probabilities = [
            [0.20, 0.80],
            [0.70, 0.30],
        ]

        return np.array(
            probabilities[:count]
        )


def test_health_returns_ok() -> None:
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
    }


def test_model_status_reports_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = (
        tmp_path
        / "rain_model.joblib"
    )

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    api_main.load_model_bundle.cache_clear()

    response = client.get(
        "/model/status"
    )

    assert response.status_code == 200

    assert response.json() == {
        "available": False,
        "model_file": "rain_model.joblib",
    }


def test_model_status_reports_available(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = (
        tmp_path
        / "rain_model.joblib"
    )

    fake_model_path.touch()

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    monkeypatch.setattr(
        api_main,
        "load_model_bundle",
        lambda: {
            "model": FakeModel(),
            "feature_columns": [
                "temperature_c",
                "humidity_percent",
                "pressure_hpa",
                "wind_speed_kmh",
            ],
            "target": "rain_next_hour",
            "forecast_horizon": "1 hour",
        },
    )

    response = client.get(
        "/model/status"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["available"] is True
    assert data["target"] == "rain_next_hour"
    assert data["forecast_horizon"] == "1 hour"


def test_predict_returns_503_when_model_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = (
        tmp_path
        / "rain_model.joblib"
    )

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    response = client.post(
        "/predict",
        json={
            "date": "2026-09-24",
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Le modèle de prédiction "
            "n'est pas disponible."
        ),
    }


def test_predict_returns_404_when_weather_data_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = (
        tmp_path
        / "rain_model.joblib"
    )

    fake_model_path.touch()

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    monkeypatch.setattr(
        api_main,
        "load_model_bundle",
        lambda: {
            "model": FakeModel(),
            "feature_columns": [
                "temperature_c",
                "humidity_percent",
                "pressure_hpa",
                "wind_speed_kmh",
            ],
            "target": "rain_next_hour",
            "forecast_horizon": "1 hour",
        },
    )

    monkeypatch.setattr(
        api_main,
        "get_weather_observations",
        lambda station_code=None: [],
    )

    response = client.post(
        "/predict",
        json={
            "date": "2026-09-24",
        },
    )

    assert response.status_code == 404

    assert (
        "Aucune observation météorologique"
        in response.json()["detail"]
    )


def test_predict_returns_real_prediction_structure(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = (
        tmp_path
        / "rain_model.joblib"
    )

    fake_model_path.touch()

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    monkeypatch.setattr(
        api_main,
        "load_model_bundle",
        lambda: {
            "model": FakeModel(),
            "feature_columns": [
                "temperature_c",
                "humidity_percent",
                "pressure_hpa",
                "wind_speed_kmh",
            ],
            "target": "rain_next_hour",
            "forecast_horizon": "1 hour",
        },
    )

    observations = [
        {
            "station_code": "000EW",
            "observed_at": "2026-09-24T00:00:00",
            "temperature_c": 18.0,
            "humidity_percent": 85.0,
            "pressure_hpa": 1014.0,
            "wind_speed_kmh": 10.0,
        },
        {
            "station_code": "000EW",
            "observed_at": "2026-09-24T01:00:00",
            "temperature_c": 18.5,
            "humidity_percent": 80.0,
            "pressure_hpa": 1015.0,
            "wind_speed_kmh": 12.0,
        },
    ]

    monkeypatch.setattr(
        api_main,
        "get_weather_observations",
        lambda station_code=None: observations,
    )

    response = client.post(
        "/predict",
        json={
            "date": "2026-09-24",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["date"] == "2026-09-24"
    assert data["station_code"] == "000EW"
    assert data["forecast_horizon"] == "1 hour"
    assert data["decision_threshold"] == 0.50

    assert data["observations_used"] == 2
    assert data["rain_alert"] is True
    assert data["hours_above_threshold"] == 1

    assert data["rain_score_max"] == 0.8
    assert data["rain_score_mean"] == 0.55

    assert len(data["hourly"]) == 2

    assert (
        data["hourly"][0]["rain_alert"]
        is True
    )

    assert (
        data["hourly"][1]["rain_alert"]
        is False
    )


def test_predict_rejects_invalid_date() -> None:
    response = client.post(
        "/predict",
        json={
            "date": "date-invalide",
        },
    )

    assert response.status_code == 422