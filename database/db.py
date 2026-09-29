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


def main() -> None:
    """Initialise la base SQLite du projet."""

    initialize_database()
    print(f"Base SQLite initialisée : {DATABASE_PATH}")


if __name__ == "__main__":
    main()