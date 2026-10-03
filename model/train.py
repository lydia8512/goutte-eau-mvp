from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "rain_model.joblib"


def load_processed_data(
    file_name: str,
) -> pd.DataFrame:
    """Charge un jeu de données préparé depuis data/processed."""

    file_path = PROCESSED_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Le fichier de données préparées est introuvable : {file_path}"
        )

    return pd.read_csv(file_path)


def validate_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:
    """Vérifie que le jeu de données est exploitable avant entraînement."""

    if dataframe.empty:
        raise ValueError(
            "Le jeu de données est vide."
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            f"La variable cible '{target_column}' "
            "est absente du jeu de données."
        )

    if dataframe[target_column].isna().any():
        raise ValueError(
            f"La variable cible '{target_column}' "
            "contient des valeurs manquantes."
        )


def train_model(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rain_next_hour",
) -> tuple[
    Pipeline,
    dict[str, float],
    list[list[int]],
]:
    """
    Entraîne et évalue un modèle léger de prévision de pluie.

    Le découpage est chronologique :
    80 % des observations les plus anciennes servent à l'entraînement
    et 20 % des observations les plus récentes servent au test.
    """

    validate_dataset(
        dataframe,
        target_column,
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Variables explicatives absentes du jeu de données : "
            + ", ".join(missing_features)
        )

    working_dataframe = dataframe.copy()

    if "observed_at" in working_dataframe.columns:
        working_dataframe["observed_at"] = pd.to_datetime(
            working_dataframe["observed_at"],
            errors="coerce",
        )

        working_dataframe = (
            working_dataframe
            .dropna(subset=["observed_at"])
            .sort_values("observed_at")
            .reset_index(drop=True)
        )

    X = working_dataframe[feature_columns]
    y = working_dataframe[target_column]

    if y.nunique() < 2:
        raise ValueError(
            "La variable cible doit contenir au moins deux classes."
        )

    split_index = int(
        len(working_dataframe) * 0.80
    )

    if split_index <= 0 or split_index >= len(
        working_dataframe
    ):
        raise ValueError(
            "Le jeu de données ne permet pas "
            "un découpage entraînement/test valide."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError(
            "Le jeu d'entraînement doit contenir "
            "au moins deux classes."
        )

    pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "standardisation",
                StandardScaler(),
            ),
            (
                "classification",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "average_precision": average_precision_score(
            y_test,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
    ).tolist()

    return (
        pipeline,
        metrics,
        matrix,
    )


def save_model(
    model: Pipeline,
    feature_columns: list[str],
    output_path: Path = MODEL_PATH,
) -> Path:
    """Sauvegarde le modèle et les variables utilisées."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": "rain_next_hour",
            "forecast_horizon": "1 hour",
        },
        output_path,
    )

    return output_path


def main() -> None:
    """Entraîne et sauvegarde le modèle réel du MVP."""

    dataframe = load_processed_data(
        "weather_dataset.csv"
    )

    feature_columns = [
        "temperature_c",
        "humidity_percent",
        "pressure_hpa",
        "wind_speed_kmh",
    ]

    model, metrics, matrix = train_model(
        dataframe=dataframe,
        feature_columns=feature_columns,
        target_column="rain_next_hour",
    )

    print("Entraînement terminé.")
    print("Métriques :")

    for name, value in metrics.items():
        print(
            f"- {name} : {value:.4f}"
        )

    print(
        f"Matrice de confusion : {matrix}"
    )

    model_path = save_model(
        model=model,
        feature_columns=feature_columns,
    )

    print(
        f"Modèle sauvegardé : {model_path}"
    )


if __name__ == "__main__":
    main()