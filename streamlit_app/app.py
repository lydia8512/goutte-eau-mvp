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


st.set_page_config(
    page_title="Goutte d'Eau – MVP",
    page_icon="🌧️",
    layout="centered",
)

st.title("Goutte d'Eau")
st.subheader("Estimation du risque de pluie")

st.write(
    "Ce MVP permet d'exploiter des données météorologiques "
    "afin d'estimer le risque de pluie sur le périmètre pilote."
)

st.info("Zone pilote : Paris – Île-de-France")

selected_date = st.date_input(
    "Date à analyser",
    value=date.today(),
)

st.write(f"Date sélectionnée : {selected_date.strftime('%d/%m/%Y')}")

if st.button("Vérifier le service API"):
    if check_api_health():
        st.success("Le service de prédiction est disponible.")
    else:
        st.warning(
            "L'API FastAPI n'est pas actuellement démarrée."
        )

st.divider()

st.caption(
    "La fonction de prédiction sera activée après l'intégration "
    "du modèle de Machine Learning."
)