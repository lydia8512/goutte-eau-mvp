from database import db


def test_initialize_database_creates_weather_table(
    tmp_path,
    monkeypatch,
):
    """Vérifie que l'initialisation crée la table météo."""

    database_path = tmp_path / "test_goutte_eau.db"

    monkeypatch.setattr(db, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(db, "DATABASE_PATH", database_path)

    db.initialize_database()

    assert database_path.exists()

    with db.get_connection() as connection:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'weather_observations'
            """
        ).fetchone()

        assert table is not None

        columns = connection.execute(
            "PRAGMA table_info(weather_observations)"
        ).fetchall()

        column_names = {column["name"] for column in columns}

        expected_columns = {
            "id",
            "station_code",
            "observed_at",
            "temperature_c",
            "humidity_percent",
            "pressure_hpa",
            "wind_speed_kmh",
            "precipitation_mm",
            "rain",
            "source",
        }

        assert expected_columns.issubset(column_names)


def test_initialize_database_can_run_twice(
    tmp_path,
    monkeypatch,
):
    """Vérifie que l'initialisation peut être relancée sans erreur."""

    database_path = tmp_path / "test_goutte_eau.db"

    monkeypatch.setattr(db, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(db, "DATABASE_PATH", database_path)

    db.initialize_database()
    db.initialize_database()

    assert database_path.exists()