import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "goutte_eau.db"


def get_connection() -> sqlite3.Connection:
    """Ouvre une connexion vers la base SQLite du projet."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """Crée les tables nécessaires au MVP si elles n'existent pas."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_code TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                temperature_c REAL,
                humidity_percent REAL,
                pressure_hpa REAL,
                wind_speed_kmh REAL,
                precipitation_mm REAL,
                rain INTEGER,
                source TEXT NOT NULL DEFAULT 'infoclimat',
                UNIQUE(station_code, observed_at)
            )
            """
        )

        connection.commit()


def insert_weather_observation(
    station_code: str,
    observed_at: str,
    temperature_c: float | None = None,
    humidity_percent: float | None = None,
    pressure_hpa: float | None = None,
    wind_speed_kmh: float | None = None,
    precipitation_mm: float | None = None,
    rain: int | None = None,
    source: str = "infoclimat",
) -> None:
    """Ajoute ou met à jour une observation météorologique."""

    initialize_database()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO weather_observations (
                station_code,
                observed_at,
                temperature_c,
                humidity_percent,
                pressure_hpa,
                wind_speed_kmh,
                precipitation_mm,
                rain,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_code, observed_at)
            DO UPDATE SET
                temperature_c = excluded.temperature_c,
                humidity_percent = excluded.humidity_percent,
                pressure_hpa = excluded.pressure_hpa,
                wind_speed_kmh = excluded.wind_speed_kmh,
                precipitation_mm = excluded.precipitation_mm,
                rain = excluded.rain,
                source = excluded.source
            """,
            (
                station_code,
                observed_at,
                temperature_c,
                humidity_percent,
                pressure_hpa,
                wind_speed_kmh,
                precipitation_mm,
                rain,
                source,
            ),
        )

        connection.commit()


def get_weather_observations(
    station_code: str | None = None,
) -> list[dict]:
    """Retourne les observations météo enregistrées dans SQLite."""

    initialize_database()

    with get_connection() as connection:
        if station_code:
            rows = connection.execute(
                """
                SELECT *
                FROM weather_observations
                WHERE station_code = ?
                ORDER BY observed_at
                """,
                (station_code,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM weather_observations
                ORDER BY observed_at
                """
            ).fetchall()

    return [dict(row) for row in rows]


def main() -> None:
    """Initialise la base SQLite du projet."""

    initialize_database()
    print(f"Base SQLite initialisée : {DATABASE_PATH}")


if __name__ == "__main__":
    main()