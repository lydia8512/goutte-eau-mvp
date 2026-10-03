from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from database.db import get_weather_observations


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "rain_model.joblib"

STATION_CODE = "000EW"
STATION_NAME = "Paris 20ème - Porte de Vincennes"

DECISION_THRESHOLD = 0.50


class PredictionRequest(BaseModel):
    """Données reçues pour une demande de prédiction."""

    date: date


app = FastAPI(
    title="Goutte d'Eau API",
    description=(
        "API du MVP Goutte d'Eau - "
        "estimation du risque de pluie à une heure."
    ),
    version="0.2.0",
)


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    """Charge le modèle entraîné et ses métadonnées."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Le modèle de prédiction n'est pas disponible."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Le fichier du modèle est invalide."
        )

    if "model" not in bundle:
        raise ValueError(
            "Le modèle entraîné est absent du fichier."
        )

    if "feature_columns" not in bundle:
        raise ValueError(
            "La liste des variables du modèle est absente."
        )

    return bundle


@app.get("/health")
def health_check() -> dict:
    """Vérifie que l'API est disponible."""

    return {
        "status": "ok",
    }


@app.get("/model/status")
def model_status() -> dict:
    """Indique si un modèle entraîné est disponible."""

    if not MODEL_PATH.exists():
        return {
            "available": False,
            "model_file": MODEL_PATH.name,
        }

    try:
        bundle = load_model_bundle()
    except Exception:
        return {
            "available": False,
            "model_file": MODEL_PATH.name,
        }

    return {
        "available": True,
        "model_file": MODEL_PATH.name,
        "target": bundle.get(
            "target",
            "rain_next_hour",
        ),
        "forecast_horizon": bundle.get(
            "forecast_horizon",
            "1 hour",
        ),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    """Estime le risque de pluie à une heure pour la date demandée."""

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Le modèle de prédiction "
                "n'est pas disponible."
            ),
        )

    try:
        bundle = load_model_bundle()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Le modèle de prédiction "
                "n'a pas pu être chargé."
            ),
        ) from error

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    observations = get_weather_observations(
        station_code=STATION_CODE,
    )

    selected_rows = []
    forecast_times = []

    for observation in observations:
        try:
            observed_at = datetime.fromisoformat(
                observation["observed_at"]
            )
        except (TypeError, ValueError):
            continue

        forecast_for = observed_at + timedelta(hours=1)

        if forecast_for.date() != request.date:
            continue

        observation_data = dict(observation)

        feature_values = {
            feature: observation_data.get(feature)
            for feature in feature_columns
        }

        if any(
            value is None
            for value in feature_values.values()
        ):
            continue

        selected_rows.append(
            feature_values
        )

        forecast_times.append(
            forecast_for
        )

    if not selected_rows:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune observation météorologique "
                "exploitable n'est disponible pour "
                "produire une estimation à cette date."
            ),
        )

    feature_dataframe = pd.DataFrame(
        selected_rows,
        columns=feature_columns,
    )

    scores = model.predict_proba(
        feature_dataframe
    )[:, 1]

    alerts = scores >= DECISION_THRESHOLD

    hourly_results = []

    for forecast_for, score, alert in zip(
        forecast_times,
        scores,
        alerts,
    ):
        hourly_results.append(
            {
                "forecast_for": forecast_for.isoformat(),
                "rain_score": round(
                    float(score),
                    4,
                ),
                "rain_alert": bool(alert),
            }
        )

    return {
        "date": request.date.isoformat(),
        "station_code": STATION_CODE,
        "station_name": STATION_NAME,
        "forecast_horizon": bundle.get(
            "forecast_horizon",
            "1 hour",
        ),
        "decision_threshold": DECISION_THRESHOLD,
        "observations_used": len(selected_rows),
        "rain_alert": bool(alerts.any()),
        "hours_above_threshold": int(alerts.sum()),
        "rain_score_max": round(
            float(scores.max()),
            4,
        ),
        "rain_score_mean": round(
            float(scores.mean()),
            4,
        ),
        "score_note": (
            "Le score est produit par le modèle "
            "de classification du MVP et ne constitue "
            "pas une probabilité météorologique calibrée."
        ),
        "hourly": hourly_results,
    }