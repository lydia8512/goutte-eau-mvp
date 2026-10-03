import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_STATION = "000EW"
OUTPUT_FILE = PROCESSED_DATA_DIR / "weather_dataset.csv"


def to_float(value) -> float | None:
    """Convertit une valeur météo en nombre décimal."""

    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_raw_observations(
    station_code: str = DEFAULT_STATION,
) -> list[dict]:
    """Charge et déduplique les observations brutes Infoclimat."""

    raw_files = sorted(
        RAW_DATA_DIR.glob(
            f"infoclimat_{station_code}_*.json"
        )
    )

    if not raw_files:
        raise FileNotFoundError(
            f"Aucun fichier brut trouvé pour la station {station_code}."
        )

    observations = {}

    for raw_file in raw_files:
        with raw_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        hourly = data.get("hourly", {})
        station_observations = hourly.get(
            station_code,
            [],
        )

        for observation in station_observations:
            observed_at = observation.get("dh_utc")

            if not observed_at:
                continue

            observations[
                (
                    station_code,
                    observed_at,
                )
            ] = observation

    return list(observations.values())


def build_dataset(
    observations: list[dict],
    station_code: str = DEFAULT_STATION,
) -> pd.DataFrame:
    """
    Transforme les données brutes en dataset de prévision.

    La cible rain_next_hour indique si de la pluie est observée
    exactement une heure après les variables météorologiques utilisées.
    """

    rows = []

    for observation in observations:
        precipitation = to_float(
            observation.get("pluie_1h")
        )

        if precipitation is None:
            continue

        temperature = to_float(
            observation.get("temperature")
        )
        pressure = to_float(
            observation.get("pression")
        )
        humidity = to_float(
            observation.get("humidite")
        )
        wind_speed = to_float(
            observation.get("vent_moyen")
        )

        if any(
            value is None
            for value in (
                temperature,
                pressure,
                humidity,
                wind_speed,
            )
        ):
            continue

        rows.append(
            {
                "station_code": station_code,
                "observed_at": observation["dh_utc"],
                "temperature_c": temperature,
                "humidity_percent": humidity,
                "pressure_hpa": pressure,
                "wind_speed_kmh": wind_speed,
                "precipitation_mm": precipitation,
                "rain": int(precipitation > 0),
            }
        )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError(
            "Aucune observation exploitable n'a été trouvée."
        )

    dataframe["observed_at"] = pd.to_datetime(
        dataframe["observed_at"],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(subset=["observed_at"])
        .drop_duplicates(
            subset=[
                "station_code",
                "observed_at",
            ]
        )
        .sort_values(
            [
                "station_code",
                "observed_at",
            ]
        )
        .reset_index(drop=True)
    )

    dataframe["next_observed_at"] = (
        dataframe
        .groupby("station_code")["observed_at"]
        .shift(-1)
    )

    dataframe["rain_next_hour"] = (
        dataframe
        .groupby("station_code")["rain"]
        .shift(-1)
    )

    dataframe["expected_next_observation"] = (
        dataframe["observed_at"]
        + pd.Timedelta(hours=1)
    )

    dataframe = dataframe[
        dataframe["next_observed_at"]
        == dataframe["expected_next_observation"]
    ].copy()

    dataframe["rain_next_hour"] = (
        dataframe["rain_next_hour"].astype(int)
    )

    dataframe["forecast_for"] = (
        dataframe["next_observed_at"]
    )

    dataframe = dataframe.drop(
        columns=[
            "next_observed_at",
            "expected_next_observation",
        ]
    )

    dataframe = dataframe.reset_index(drop=True)

    return dataframe


def save_dataset(
    dataframe: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> Path:
    """Enregistre le dataset nettoyé au format CSV."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    return output_file


def main() -> None:
    """Construit le dataset de prévision exploitable par le modèle."""

    observations = load_raw_observations()

    dataframe = build_dataset(
        observations=observations,
    )

    output_file = save_dataset(
        dataframe=dataframe,
    )

    rain_count = int(
        dataframe["rain_next_hour"].sum()
    )

    no_rain_count = int(
        (dataframe["rain_next_hour"] == 0).sum()
    )

    print("Préparation du dataset de prévision terminée.")
    print(
        f"Observations brutes uniques : "
        f"{len(observations)}"
    )
    print(
        f"Lignes avec horizon exact à 1 heure : "
        f"{len(dataframe)}"
    )
    print(
        f"Sans pluie à l'heure suivante : "
        f"{no_rain_count}"
    )
    print(
        f"Avec pluie à l'heure suivante : "
        f"{rain_count}"
    )
    print(
        f"Fichier créé : {output_file}"
    )


if __name__ == "__main__":
    main()