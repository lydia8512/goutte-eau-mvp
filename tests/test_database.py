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


def test_insert_weather_observation(
    tmp_path,
    monkeypatch,
):
    """Vérifie qu'une observation météo peut être enregistrée."""

    database_path = tmp_path / "test_goutte_eau.db"

    monkeypatch.setattr(db, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(db, "DATABASE_PATH", database_path)

    db.insert_weather_observation(
        station_code="TEST001",
        observed_at="2026-09-29T12:00:00",
        temperature_c=18.5,
        humidity_percent=75.0,
        pressure_hpa=1015.2,
        wind_speed_kmh=12.0,
        precipitation_mm=1.4,
        rain=1,
    )

    with db.get_connection() as connection:
        observation = connection.execute(
            """
            SELECT *
            FROM weather_observations
            WHERE station_code = ?
            AND observed_at = ?
            """,
            (
                "TEST001",
                "2026-09-29T12:00:00",
            ),
        ).fetchone()

    assert observation is not None
    assert observation["temperature_c"] == 18.5
    assert observation["humidity_percent"] == 75.0
    assert observation["precipitation_mm"] == 1.4
    assert observation["rain"] == 1
    assert observation["source"] == "infoclimat"


def test_insert_weather_observation_updates_existing_row(
    tmp_path,
    monkeypatch,
):
    """Vérifie qu'une observation existante est mise à jour."""

    database_path = tmp_path / "test_goutte_eau.db"

    monkeypatch.setattr(db, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(db, "DATABASE_PATH", database_path)

    db.insert_weather_observation(
        station_code="TEST001",
        observed_at="2026-09-29T12:00:00",
        temperature_c=18.5,
        rain=0,
    )

    db.insert_weather_observation(
        station_code="TEST001",
        observed_at="2026-09-29T12:00:00",
        temperature_c=19.2,
        rain=1,
    )

    with db.get_connection() as connection:
        observations = connection.execute(
            """
            SELECT *
            FROM weather_observations
            WHERE station_code = ?
            AND observed_at = ?
            """,
            (
                "TEST001",
                "2026-09-29T12:00:00",
            ),
        ).fetchall()

    assert len(observations) == 1
    assert observations[0]["temperature_c"] == 19.2
    assert observations[0]["rain"] == 1