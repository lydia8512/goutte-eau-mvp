from datetime import date

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"

DEFAULT_DEMO_DATE = date(2026, 9, 24)


def check_api_health() -> bool:
    """Vérifie si l'API FastAPI est disponible."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_model_status() -> dict | None:
    """Récupère l'état du modèle de Machine Learning."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/model/status",
            timeout=3,
        )

        if response.status_code != 200:
            return None

        return response.json()

    except requests.RequestException:
        return None


def request_prediction(
    selected_date: date,
) -> tuple[int, dict]:
    """Envoie une demande de prédiction à FastAPI."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "date": selected_date.isoformat(),
            },
            timeout=10,
        )

        try:
            response_data = response.json()

        except ValueError:
            response_data = {
                "detail": (
                    "La réponse reçue par l'API "
                    "n'est pas exploitable."
                )
            }

        return (
            response.status_code,
            response_data,
        )

    except requests.RequestException:
        return (
            0,
            {
                "detail": (
                    "Impossible de communiquer "
                    "avec le service de prédiction."
                )
            },
        )


st.set_page_config(
    page_title="Goutte d'Eau - MVP",
    page_icon="💧",
    layout="wide",
)

st.title("Goutte d'Eau")

st.subheader(
    "Estimation du risque de pluie à court terme"
)

st.write(
    "Ce MVP exploite des observations météorologiques "
    "et un modèle de Machine Learning afin d'estimer "
    "le risque de pluie à l'horizon d'une heure."
)

st.info(
    "Zone pilote : Paris 20ème - Porte de Vincennes "
    "(station 000EW)"
)

st.caption(
    "La date du 24/09/2026 est proposée par défaut "
    "afin de permettre une démonstration reproductible "
    "avec les données disponibles dans le MVP."
)

selected_date = st.date_input(
    "Date à analyser",
    value=DEFAULT_DEMO_DATE,
)

st.write(
    "Date sélectionnée : "
    f"{selected_date.strftime('%d/%m/%Y')}"
)

st.divider()

st.subheader("État du service")

if st.button(
    "Vérifier le service API",
    use_container_width=True,
):
    if check_api_health():
        st.success(
            "Le service FastAPI est disponible."
        )

        model_status = get_model_status()

        if model_status is None:
            st.warning(
                "L'état du modèle n'a pas pu "
                "être récupéré."
            )

        elif model_status.get("available"):
            st.success(
                "Le modèle de Machine Learning "
                "est chargé et disponible."
            )

            st.write(
                "Horizon de prévision : "
                f"{model_status.get('forecast_horizon', 'Non précisé')}"
            )

        else:
            st.warning(
                "L'API fonctionne mais le modèle "
                "de Machine Learning n'est pas disponible."
            )

    else:
        st.error(
            "L'API FastAPI n'est pas disponible."
        )

st.divider()

st.subheader("Estimation du risque de pluie")

if st.button(
    "Estimer le risque de pluie",
    type="primary",
    use_container_width=True,
):
    if not check_api_health():
        st.error(
            "Impossible d'effectuer l'estimation : "
            "le service FastAPI n'est pas disponible."
        )

    else:
        status_code, prediction_data = (
            request_prediction(
                selected_date
            )
        )

        if status_code == 200:
            rain_alert = prediction_data.get(
                "rain_alert",
                False,
            )

            maximum_score = prediction_data.get(
                "rain_score_max",
                0,
            )

            mean_score = prediction_data.get(
                "rain_score_mean",
                0,
            )

            hours_above_threshold = (
                prediction_data.get(
                    "hours_above_threshold",
                    0,
                )
            )

            observations_used = (
                prediction_data.get(
                    "observations_used",
                    0,
                )
            )

            decision_threshold = (
                prediction_data.get(
                    "decision_threshold",
                    0.5,
                )
            )

            if rain_alert:
                st.warning(
                    "Risque de pluie détecté : "
                    "au moins une heure dépasse "
                    "le seuil de décision du modèle."
                )
            else:
                st.success(
                    "Aucune heure ne dépasse "
                    "le seuil de décision du modèle."
                )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Score maximal",
                    f"{maximum_score:.1%}",
                )

            with col2:
                st.metric(
                    "Score moyen",
                    f"{mean_score:.1%}",
                )

            with col3:
                st.metric(
                    "Heures en alerte",
                    hours_above_threshold,
                )

            with col4:
                st.metric(
                    "Observations utilisées",
                    observations_used,
                )

            st.write(
                "Seuil de décision : "
                f"{decision_threshold:.0%}"
            )

            st.write(
                "Horizon de prévision : "
                f"{prediction_data.get('forecast_horizon', 'Non précisé')}"
            )

            st.write(
                "Station : "
                f"{prediction_data.get('station_name', 'Non précisée')}"
            )

            hourly_data = prediction_data.get(
                "hourly",
                [],
            )

            if hourly_data:
                dataframe = pd.DataFrame(
                    hourly_data
                )

                dataframe[
                    "forecast_for"
                ] = pd.to_datetime(
                    dataframe["forecast_for"]
                )

                dataframe[
                    "heure"
                ] = dataframe[
                    "forecast_for"
                ].dt.strftime("%H:%M")

                dataframe[
                    "score_risque"
                ] = (
                    dataframe[
                        "rain_score"
                    ]
                    * 100
                ).round(1)

                dataframe[
                    "alerte"
                ] = dataframe[
                    "rain_alert"
                ].map(
                    {
                        True: "Oui",
                        False: "Non",
                    }
                )

                st.subheader(
                    "Évolution horaire du score"
                )

                chart_data = (
                    dataframe[
                        [
                            "heure",
                            "score_risque",
                        ]
                    ]
                    .set_index("heure")
                )

                st.line_chart(
                    chart_data,
                )

                st.subheader(
                    "Détail heure par heure"
                )

                display_dataframe = (
                    dataframe[
                        [
                            "heure",
                            "score_risque",
                            "alerte",
                        ]
                    ]
                    .rename(
                        columns={
                            "heure": "Heure",
                            "score_risque": (
                                "Score de risque (%)"
                            ),
                            "alerte": "Alerte",
                        }
                    )
                )

                st.dataframe(
                    display_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

            st.info(
                prediction_data.get(
                    "score_note",
                    (
                        "Le score fourni par le modèle "
                        "ne constitue pas une probabilité "
                        "météorologique calibrée."
                    ),
                )
            )

        elif status_code == 404:
            st.warning(
                prediction_data.get(
                    "detail",
                    (
                        "Aucune donnée exploitable "
                        "n'est disponible pour cette date."
                    ),
                )
            )

        elif status_code == 503:
            st.warning(
                prediction_data.get(
                    "detail",
                    (
                        "Le service de prédiction "
                        "n'est pas disponible."
                    ),
                )
            )

        elif status_code == 422:
            st.error(
                "La date transmise au service "
                "de prédiction est invalide."
            )

        else:
            st.error(
                prediction_data.get(
                    "detail",
                    (
                        "Une erreur est survenue "
                        "pendant l'estimation."
                    ),
                )
            )

st.divider()

st.caption(
    "MVP Goutte d'Eau - modèle de classification "
    "à horizon d'une heure sur le périmètre pilote."
)