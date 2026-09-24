from datetime import date

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


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
    """Récupère l'état de disponibilité du modèle."""

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


def request_prediction(selected_date: date) -> tuple[int, dict]:
    """Envoie une demande de prédiction à l'API."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"date": selected_date.isoformat()},
            timeout=5,
        )

        try:
            response_data = response.json()
        except ValueError:
            response_data = {
                "detail": "La réponse reçue par l'API n'est pas exploitable."
            }

        return response.status_code, response_data

    except requests.RequestException:
        return 0, {
            "detail": "Impossible de communiquer avec le service de prédiction."
        }


st.set_page_config(
    page_title="Goutte d'Eau - MVP",
    page_icon="💧",
    layout="centered",
)

st.title("Goutte d'Eau")
st.subheader("Estimation du risque de pluie")

st.write(
    "Ce MVP permet d'exploiter des données météorologiques "
    "afin d'estimer le risque de pluie sur le périmètre pilote."
)

st.info("Zone pilote : Paris - Île-de-France")

selected_date = st.date_input(
    "Date à analyser",
    value=date.today(),
)

st.write(
    f"Date sélectionnée : {selected_date.strftime('%d/%m/%Y')}"
)

st.divider()

st.subheader("État du service")

if st.button("Vérifier le service API"):
    if check_api_health():
        st.success("Le service de prédiction est disponible.")

        model_status = get_model_status()

        if model_status is None:
            st.warning(
                "L'état du modèle n'a pas pu être récupéré."
            )
        elif model_status.get("available"):
            st.success(
                "Le modèle de Machine Learning est disponible."
            )
        else:
            st.info(
                "Le service API fonctionne, mais le modèle de "
                "Machine Learning n'est pas encore disponible."
            )
    else:
        st.error(
            "L'API FastAPI n'est pas actuellement disponible."
        )

st.divider()

st.subheader("Prédiction")

if st.button("Estimer le risque de pluie"):
    if not check_api_health():
        st.error(
            "Impossible d'effectuer la prédiction : "
            "le service API n'est pas disponible."
        )
    else:
        status_code, prediction_data = request_prediction(
            selected_date
        )

        if status_code == 200:
            risk = prediction_data.get(
                "risque_pluie",
                prediction_data.get(
                    "prediction",
                    "Résultat disponible",
                ),
            )

            probability = prediction_data.get("probabilite")

            st.success(f"Résultat : {risk}")

            if probability is not None:
                st.metric(
                    "Probabilité estimée",
                    f"{probability:.1%}",
                )

        elif status_code == 503:
            st.warning(
                prediction_data.get(
                    "detail",
                    "Le service de prédiction n'est pas encore disponible.",
                )
            )

        elif status_code == 422:
            st.error(
                "La demande de prédiction contient une donnée invalide."
            )

        else:
            st.error(
                prediction_data.get(
                    "detail",
                    "Une erreur est survenue lors de la prédiction.",
                )
            )

st.divider()

st.caption(
    "MVP Goutte d'Eau - estimation du risque de pluie "
    "sur le périmètre pilote."
)