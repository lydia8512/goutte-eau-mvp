# Goutte d'Eau – MVP de prévision du risque de pluie

## Présentation du projet

Goutte d'Eau est un MVP visant à estimer le risque de pluie sur un périmètre pilote situé à Paris et en Île-de-France.

Le projet s'appuie sur une architecture légère associant :

- une collecte de données météorologiques ;
- une base de données SQLite ;
- un modèle de Machine Learning ;
- une API développée avec FastAPI ;
- une interface utilisateur développée avec Streamlit.

L'objectif est de disposer d'une solution simple, testable et évolutive permettant d'exploiter des données météorologiques afin de produire une estimation du risque de pluie.

## État actuel du MVP

Les composants techniques principaux du projet sont en place.

À ce stade :

- le dépôt Git est initialisé et synchronisé avec GitHub ;
- l'environnement Python est configuré ;
- l'API FastAPI est opérationnelle ;
- l'interface Streamlit est opérationnelle ;
- Streamlit communique avec FastAPI ;
- les tests automatisés sont fonctionnels ;
- le pipeline de Machine Learning est préparé ;
- la structure SQLite est préparée ;
- la collecte des données météorologiques est préparée.

L'entraînement définitif du modèle n'a pas encore été réalisé, car les données météorologiques réelles doivent encore être collectées et préparées.

Aucune prédiction artificielle n'est générée tant que le modèle réel n'est pas disponible.

## Architecture générale

Le fonctionnement prévu du MVP est le suivant :

1. récupération des données météorologiques ;
2. nettoyage et préparation des données ;
3. stockage des données utiles dans SQLite ;
4. entraînement du modèle de Machine Learning ;
5. sauvegarde du modèle entraîné ;
6. exposition du service de prédiction avec FastAPI ;
7. consultation du résultat dans l'interface Streamlit.

## Structure du projet

```text
goutte-eau-mvp/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── __init__.py
│   ├── collect_infoclimat.py
│   └── config.py
│
├── database/
│   ├── __init__.py
│   └── db.py
│
├── docs/
│
├── model/
│   ├── __init__.py
│   └── train.py
│
├── streamlit_app/
│   ├── __init__.py
│   └── app.py
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

## Technologies utilisées

Le MVP utilise principalement :

- Python ;
- FastAPI ;
- Uvicorn ;
- Streamlit ;
- pandas ;
- scikit-learn ;
- SQLite ;
- joblib ;
- pytest ;
- requests ;
- Git ;
- GitHub.

## Installation

Cloner le dépôt puis se placer dans le dossier du projet.

Créer et activer un environnement virtuel Python.

Sous Windows :

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Installer ensuite les dépendances :

    pip install -r requirements.txt

## Configuration

Les variables sensibles sont stockées localement dans un fichier `.env`.

Le fichier `.env` n'est pas versionné dans Git.

Un fichier `.env.example` est fourni pour documenter la configuration attendue.

Exemple :

    INFOCLIMAT_API_KEY=

La clé réelle ne doit jamais être publiée dans le dépôt GitHub.

## Lancement de l'API FastAPI

Depuis la racine du projet :

    python -m uvicorn api.main:app --reload

L'API est alors accessible localement à l'adresse :

    http://127.0.0.1:8000

La documentation interactive Swagger est accessible à l'adresse :

    http://127.0.0.1:8000/docs

## Routes disponibles

### GET /health

Permet de vérifier que l'API est disponible.

Exemple de réponse :

    {
      "status": "ok"
    }

### GET /model/status

Permet de vérifier si un modèle entraîné est disponible.

À ce stade, tant que le modèle n'a pas été entraîné, la réponse indique :

    {
      "available": false,
      "model_file": "rain_model.joblib"
    }

### POST /predict

Cette route reçoit une date au format JSON.

Exemple :

    {
      "date": "2026-09-24"
    }

Tant que le modèle réel n'est pas disponible, l'API retourne volontairement une réponse HTTP 503.

Ce comportement évite de produire une prédiction fictive ou non fondée sur des données réelles.

## Lancement de l'interface Streamlit

Dans un second terminal, depuis la racine du projet :

    python -m streamlit run streamlit_app/app.py

L'interface est ensuite accessible à l'adresse :

    http://localhost:8501

L'utilisateur peut :

- sélectionner une date ;
- vérifier l'état de l'API ;
- vérifier la disponibilité du modèle ;
- lancer une demande de prédiction.

Lorsque le modèle n'est pas encore disponible, l'interface affiche un message explicite au lieu de présenter un résultat artificiel.

## Tests automatisés

Les tests peuvent être exécutés avec :

    python -m pytest -q

Ils couvrent notamment :

- la disponibilité de l'API ;
- le statut du modèle ;
- le comportement de la route de prédiction ;
- la validation des dates ;
- la validation du dataset ;
- le pipeline d'entraînement ;
- la sauvegarde du modèle.

## Machine Learning

Le pipeline de Machine Learning est préparé dans :

    model/train.py

Il comprend notamment :

- la validation du dataset ;
- la sélection des variables ;
- la séparation entraînement / test ;
- la gestion des valeurs manquantes ;
- la standardisation des variables ;
- un modèle de régression logistique ;
- le calcul d'indicateurs de performance ;
- la sauvegarde du modèle entraîné.

Le choix d'un modèle léger permet de conserver une architecture adaptée à un MVP et de limiter la consommation de ressources.

## Données météorologiques

Les données réelles n'ont pas encore été intégrées au modèle définitif.

Le projet prévoit leur collecte, leur nettoyage puis leur stockage avant entraînement.

La connexion à la source de données sera activée lorsque la configuration de la clé API sera finalisée.

## Limites actuelles

À ce stade, le MVP ne produit pas encore de véritable estimation météorologique.

Cette limitation est volontaire : une prédiction ne sera activée qu'après :

- récupération de données météorologiques réelles ;
- contrôle de leur qualité ;
- constitution du dataset ;
- entraînement et évaluation du modèle ;
- intégration des variables nécessaires à l'inférence.

## Prochaines étapes

Les prochaines étapes techniques sont :

1. connecter la source de données météorologiques ;
2. récupérer les données réelles ;
3. nettoyer et préparer le dataset ;
4. alimenter la base SQLite ;
5. entraîner le modèle ;
6. analyser les métriques de performance ;
7. sauvegarder le modèle validé ;
8. connecter réellement la route `/predict` ;
9. afficher le résultat dans Streamlit ;
10. compléter la documentation technique avec les résultats obtenus.

## Qualité et reproductibilité

Le projet repose sur :

- un environnement virtuel Python ;
- un fichier `requirements.txt` ;
- un versionnement Git ;
- des tests automatisés ;
- une séparation des composants ;
- une gestion séparée des informations sensibles ;
- une documentation des commandes nécessaires au lancement du MVP.

Ces choix facilitent la maintenance, la reproductibilité et l'évolution progressive de la solution.