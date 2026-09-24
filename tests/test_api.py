from fastapi.testclient import TestClient

import api.main as api_main


client = TestClient(api_main.app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_status_reports_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = tmp_path / "rain_model.joblib"

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    response = client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "model_file": "rain_model.joblib",
    }


def test_predict_returns_503_when_model_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = tmp_path / "rain_model.joblib"

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    response = client.post(
        "/predict",
        json={"date": "2026-09-24"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Le modèle de prédiction n'est pas encore disponible."
    }


def test_predict_returns_503_when_weather_data_is_not_connected(
    monkeypatch,
    tmp_path,
) -> None:
    fake_model_path = tmp_path / "rain_model.joblib"
    fake_model_path.touch()

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    response = client.post(
        "/predict",
        json={"date": "2026-09-24"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Le modèle est disponible, mais l'alimentation des variables "
            "météorologiques par date n'est pas encore activée."
        )
    }


def test_predict_rejects_invalid_date() -> None:
    response = client.post(
        "/predict",
        json={"date": "date-invalide"},
    )

    assert response.status_code == 422