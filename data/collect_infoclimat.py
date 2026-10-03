from datetime import date, timedelta
from pathlib import Path
import json
import time

import requests

from data.config import get_infoclimat_api_key


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

INFOCLIMAT_URL = "https://www.infoclimat.fr/opendata/"

DEFAULT_STATION = "000EW"
DEFAULT_STATION_NAME = "Paris 20ème - Porte de Vincennes"

MAX_PERIOD_DAYS = 7


def save_raw_data(
    data: dict,
    station_code: str,
    start_date: date,
    end_date: date,
) -> Path:
    """Enregistre une réponse brute Infoclimat dans data/raw."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        RAW_DATA_DIR
        / f"infoclimat_{station_code}_{start_date}_{end_date}.json"
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_file


def fetch_infoclimat_data(
    station_code: str,
    start_date: date,
    end_date: date,
) -> dict:
    """Récupère les observations météo d'une station Infoclimat."""

    if end_date < start_date:
        raise ValueError(
            "La date de fin doit être postérieure ou égale à la date de début."
        )

    period_days = (end_date - start_date).days + 1

    if period_days > MAX_PERIOD_DAYS:
        raise ValueError(
            "Une requête Infoclimat ne doit pas dépasser "
            f"{MAX_PERIOD_DAYS} jours calendaires."
        )

    api_key = get_infoclimat_api_key()

    response = requests.get(
        INFOCLIMAT_URL,
        params=[
            ("method", "get"),
            ("format", "json"),
            ("stations[]", station_code),
            ("start", start_date.isoformat()),
            ("end", end_date.isoformat()),
            ("token", api_key),
        ],
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "OK":
        raise RuntimeError(
            f"Erreur Infoclimat : {data.get('errors', [])}"
        )

    hourly = data.get("hourly", {})

    if station_code not in hourly:
        raise RuntimeError(
            f"Aucune donnée horaire disponible pour la station {station_code}."
        )

    return data


def collect_period(
    station_code: str,
    start_date: date,
    end_date: date,
) -> Path:
    """Collecte puis sauvegarde les données brutes d'une période."""

    data = fetch_infoclimat_data(
        station_code=station_code,
        start_date=start_date,
        end_date=end_date,
    )

    return save_raw_data(
        data=data,
        station_code=station_code,
        start_date=start_date,
        end_date=end_date,
    )


def split_period(
    start_date: date,
    end_date: date,
) -> list[tuple[date, date]]:
    """Découpe une période en blocs de sept jours maximum."""

    if end_date < start_date:
        raise ValueError(
            "La date de fin doit être postérieure ou égale à la date de début."
        )

    periods = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(
            current_start + timedelta(days=MAX_PERIOD_DAYS - 1),
            end_date,
        )

        periods.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = current_end + timedelta(days=1)

    return periods


def collect_historical_period(
    station_code: str,
    start_date: date,
    end_date: date,
    pause_seconds: float = 1.0,
) -> list[Path]:
    """Collecte une période longue en plusieurs requêtes Infoclimat."""

    periods = split_period(
        start_date=start_date,
        end_date=end_date,
    )

    output_files = []

    for index, (period_start, period_end) in enumerate(
        periods,
        start=1,
    ):
        print(
            f"Collecte {index}/{len(periods)} : "
            f"{period_start} -> {period_end}"
        )

        output_file = collect_period(
            station_code=station_code,
            start_date=period_start,
            end_date=period_end,
        )

        output_files.append(output_file)

        if index < len(periods):
            time.sleep(pause_seconds)

    return output_files


def main() -> None:
    """Exécute une collecte de contrôle sur la station pilote."""

    start_date = date(2026, 9, 27)
    end_date = date(2026, 10, 3)

    output_file = collect_period(
        station_code=DEFAULT_STATION,
        start_date=start_date,
        end_date=end_date,
    )

    print("Collecte Infoclimat terminée.")
    print(
        f"Station : {DEFAULT_STATION} - "
        f"{DEFAULT_STATION_NAME}"
    )
    print(f"Fichier créé : {output_file}")


if __name__ == "__main__":
    main()