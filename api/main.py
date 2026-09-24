from fastapi import FastAPI


app = FastAPI(
    title="Goutte d'Eau API",
    description="API du MVP Goutte d'Eau.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    """Vérifie que l'API est disponible."""

    return {"status": "ok"}