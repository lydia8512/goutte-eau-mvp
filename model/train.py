from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "rain_model.joblib"


def load_processed_data(file_name: str) -> pd.DataFrame:
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
        raise ValueError("Le jeu de données est vide.")

    if target_column not in dataframe.columns:
        raise ValueError(
            f"La variable cible '{target_column}' est absente du jeu de données."
        )

    if dataframe[target_column].isna().any():
        raise ValueError(
            f"La variable cible '{target_column}' contient des valeurs manquantes."
        )


def train_model(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rain",
) -> tuple[Pipeline, dict[str, float], list[list[int]]]:
    """Entraîne et évalue un modèle léger de classification du risque de pluie."""

    validate_dataset(dataframe, target_column)

    missing_features = [
        column for column in feature_columns if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Variables explicatives absentes du jeu de données : "
            + ", ".join(missing_features)
        )

    X = dataframe[feature_columns]
    y = dataframe[target_column]

    if y.nunique() < 2:
        raise ValueError(
            "La variable cible doit contenir au moins deux classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("standardisation", StandardScaler()),
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

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
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
    }

    matrix = confusion_matrix(y_test, predictions).tolist()

    return pipeline, metrics, matrix


def save_model(
    model: Pipeline,
    feature_columns: list[str],
    output_path: Path = MODEL_PATH,
) -> Path:
    """Sauvegarde le modèle entraîné et la liste des variables utilisées."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        output_path,
    )

    return output_path


def main() -> None:
    print(
        "Le pipeline Machine Learning est prêt. "
        "L'entraînement définitif sera exécuté après préparation "
        "des données météorologiques réelles."
    )


if __name__ == "__main__":
    main()