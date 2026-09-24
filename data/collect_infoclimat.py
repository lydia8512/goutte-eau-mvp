from datetime import datetime
from pathlib import Path
import json

from data.config import get_infoclimat_api_key


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def save_raw_data(data: dict, prefix: str = "infoclimat") -> Path:
    """Enregistre une réponse brute de l'API dans le dossier data/raw."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RAW_DATA_DIR / f"{prefix}_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return output_file


def main() -> None:
    """Point d'entrée du script de collecte."""

    api_key = get_infoclimat_api_key()

    print("Clé API Infoclimat détectée.")
    print(f"Dossier de stockage : {RAW_DATA_DIR}")
    print("La requête API sera configurée lors de l'activation de l'accès Infoclimat.")


if __name__ == "__main__":
    main()