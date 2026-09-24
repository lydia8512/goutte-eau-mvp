from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "rain_model.joblib"


class PredictionRequest(BaseModel):
    """Données reçues pour une demande de prédiction."""

    date: date


app = FastAPI(
    title="Goutte d'Eau API",
    description="API du MVP Goutte d'Eau.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    """Vérifie que l'API est disponible."""

    return {"status": "ok"}


@app.get("/model/status")
def model_status() -> dict:
    """Indique si un modèle entraîné est disponible."""

    return {
        "available": MODEL_PATH.exists(),
        "model_file": MODEL_PATH.name,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    """
    Reçoit une date et prépare l'appel au service de prédiction.

    La prédiction réelle sera activée après entraînement du modèle
    et intégration des données météorologiques nécessaires.
    """

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Le modèle de prédiction n'est pas encore disponible.",
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "Le modèle est disponible, mais l'alimentation des variables "
            "météorologiques par date n'est pas encore activée."
        ),
    )