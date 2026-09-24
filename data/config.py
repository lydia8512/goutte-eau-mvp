import os

from dotenv import load_dotenv


load_dotenv()

INFOCLIMAT_API_KEY = os.getenv("INFOCLIMAT_API_KEY")


def get_infoclimat_api_key() -> str:
    if not INFOCLIMAT_API_KEY:
        raise ValueError(
            "La clé API Infoclimat est absente du fichier .env."
        )

    return INFOCLIMAT_API_KEY