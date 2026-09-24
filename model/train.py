from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model"


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


def main() -> None:
    print(
        "Le pipeline d'entraînement sera exécuté après "
        "la préparation des données météorologiques réelles."
    )


if __name__ == "__main__":
    main()