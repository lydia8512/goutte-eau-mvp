# Goutte d'Eau – MVP d'estimation du risque de pluie

## Présentation

Goutte d'Eau est un MVP destiné à exploiter des données météorologiques afin d'estimer le risque de pluie à court terme sur un périmètre pilote.

La solution repose sur une architecture légère associant une collecte de données météorologiques, un pipeline de préparation des données, une base SQLite, un modèle de Machine Learning, une API FastAPI et une interface Streamlit.

Le périmètre pilote actuellement implémenté correspond à la station météorologique Infoclimat `000EW – Paris 20ème - Porte de Vincennes`.

L'objectif du MVP est de démontrer la faisabilité technique d'une chaîne complète allant de la donnée météorologique brute jusqu'à la restitution d'un score de risque exploitable par une interface utilisateur.

---

## État du MVP

Le MVP est fonctionnel de bout en bout.

La chaîne actuellement opérationnelle est la suivante :

1. collecte de données météorologiques réelles depuis Infoclimat ;
2. stockage local des données brutes ;
3. nettoyage et préparation des observations ;
4. constitution d'un dataset de prévision à horizon d'une heure ;
5. stockage des observations nettoyées dans SQLite ;
6. entraînement d'un modèle de classification ;
7. sauvegarde du modèle entraîné ;
8. exposition du modèle avec FastAPI ;
9. appel de l'API depuis Streamlit ;
10. restitution du score de risque sous forme de synthèse, graphique et tableau horaire ;
11. validation du fonctionnement avec des tests automatisés.

Le dépôt Git est synchronisé avec GitHub et contient le code source, le dataset préparé et le modèle entraîné.

Les données brutes, la base SQLite locale et les secrets d'API restent exclus du versionnement.

---

## Architecture fonctionnelle

Le fonctionnement du MVP suit le flux suivant :

```text
Infoclimat Open Data
        │
        ▼
Collecte des observations
data/collect_infoclimat.py
        │
        ▼
Données brutes JSON
data/raw/
        │
        ▼
Nettoyage et préparation
data/prepare_dataset.py
        │
        ├──────────────► Dataset préparé
        │                weather_dataset.csv
        │
        ▼
Base SQLite
weather_observations
        │
        ▼
Entraînement Machine Learning
model/train.py
        │
        ▼
rain_model.joblib
        │
        ▼
FastAPI
POST /predict
        │
        ▼
Streamlit
Interface utilisateur
```

---

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
│   │   └── weather_dataset.csv
│   ├── __init__.py
│   ├── collect_infoclimat.py
│   ├── config.py
│   └── prepare_dataset.py
│
├── database/
│   ├── __init__.py
│   └── db.py
│
├── docs/
│
├── model/
│   ├── __init__.py
│   ├── train.py
│   └── rain_model.joblib
│
├── streamlit_app/
│   ├── __init__.py
│   └── app.py
│
├── tests/
│   ├── test_api.py
│   ├── test_database.py
│   └── test_model.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies

Le MVP utilise principalement Python, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, SQLite, joblib, requests, pytest, Git et GitHub.

Le choix de composants légers permet de conserver une architecture adaptée à un MVP, simple à maintenir et peu consommatrice de ressources.

---

## Données météorologiques

Les observations utilisées proviennent de l'API Open Data d'Infoclimat.

Station pilote :

```text
Code : 000EW
Nom : Paris 20ème - Porte de Vincennes
Réseau : StatIC
Source : Infoclimat
```

La période collectée pour le MVP s'étend du `01/10/2025` au `03/10/2026`.

La collecte a permis de récupérer plus de 52 000 observations uniques avant filtrage.

Les fichiers JSON bruts sont conservés localement dans :

```text
data/raw/
```

Ils ne sont pas versionnés dans Git afin d'éviter d'alourdir inutilement le dépôt.

Le dataset final préparé est en revanche inclus dans le dépôt :

```text
data/processed/weather_dataset.csv
```

Il contient `8 599` observations exploitables avec un horizon de prévision exactement égal à une heure.

Répartition de la variable cible :

```text
Absence de pluie à T+1 : 8 045 observations
Pluie à T+1            :   554 observations
Total                   : 8 599 observations
```

Cette répartition met en évidence un déséquilibre important entre les deux classes, pris en compte lors de l'entraînement du modèle.

---

## Préparation du dataset

Le script :

```text
data/prepare_dataset.py
```

construit le dataset de Machine Learning à partir des observations brutes.

La variable cible est :

```text
rain_next_hour
```

Elle indique si de la pluie est observée à l'heure suivante.

Seules les transitions temporelles correspondant exactement à une heure sont conservées.

Les variables utilisées par le modèle sont :

```text
temperature_c
humidity_percent
pressure_hpa
wind_speed_kmh
```

Les variables décrivant directement la pluie observée à l'instant courant ne sont pas utilisées comme variables explicatives du modèle final afin de limiter les risques de fuite d'information.

---

## Base SQLite

Les observations météorologiques nettoyées peuvent être stockées dans une base SQLite locale :

```text
database/goutte_eau.db
```

La table principale est :

```text
weather_observations
```

Elle contient notamment la station, la date et l'heure de l'observation, la température, l'humidité, la pression atmosphérique, la vitesse du vent, les précipitations observées et l'indicateur de pluie.

L'unicité d'une observation est contrôlée par le couple :

```text
station_code + observed_at
```

Les opérations d'insertion utilisent un mécanisme d'upsert afin d'éviter la création de doublons.

La base SQLite est volontairement exclue du dépôt Git car elle constitue un artefact local généré à partir des données du projet.

---

## Machine Learning

Le pipeline d'entraînement est défini dans :

```text
model/train.py
```

Le modèle retenu pour le MVP est une régression logistique.

Ce choix répond à plusieurs objectifs :

- modèle léger ;
- temps d'entraînement réduit ;
- faible consommation de ressources ;
- comportement interprétable ;
- complexité adaptée à un MVP ;
- absence de recours inutile à un modèle de Deep Learning.

Le pipeline comprend une imputation des valeurs manquantes par médiane, une standardisation des variables et une régression logistique avec pondération des classes.

La séparation entraînement / test est réalisée chronologiquement.

Les `80 %` premières observations sont utilisées pour l'entraînement et les `20 %` les plus récentes sont utilisées pour l'évaluation.

Cette approche évite une séparation aléatoire susceptible de mélanger artificiellement des observations passées et futures.

---

## Résultats du modèle

Les métriques obtenues sur le jeu de test chronologique sont :

| Indicateur | Résultat |
| --- | ---: |
| Accuracy | 0.8953 |
| Balanced accuracy | 0.7321 |
| Precision | 0.0978 |
| Recall | 0.5625 |
| F1-score | 0.1667 |
| Average precision | 0.1686 |
| ROC AUC | 0.8376 |

Matrice de confusion :

```text
[[1522, 166],
 [  14,  18]]
```

Le seuil de décision du MVP est fixé à :

```text
0.50
```

Le modèle sauvegardé est disponible dans :

```text
model/rain_model.joblib
```

La classe positive étant nettement minoritaire, l'accuracy seule n'est pas considérée comme suffisante pour évaluer le modèle. Les indicateurs de rappel, balanced accuracy, average precision et ROC AUC sont également suivis.

Ces résultats correspondent à un prototype et ne constituent pas une validation météorologique opérationnelle.

---

## API FastAPI

L'API est développée dans :

```text
api/main.py
```

Pour la lancer depuis la racine du projet :

```powershell
python -m uvicorn api.main:app --reload
```

Elle est accessible localement sur :

```text
http://127.0.0.1:8000
```

La documentation interactive Swagger est disponible sur :

```text
http://127.0.0.1:8000/docs
```

### GET /health

Vérifie que l'API est disponible.

Exemple :

```json
{
  "status": "ok"
}
```

### GET /model/status

Vérifie que le modèle entraîné est disponible.

Exemple :

```json
{
  "available": true,
  "model_file": "rain_model.joblib",
  "target": "rain_next_hour",
  "forecast_horizon": "1 hour"
}
```

### POST /predict

La route reçoit une date :

```json
{
  "date": "2026-09-24"
}
```

Pour chaque observation exploitable permettant de produire une estimation à cette date, le modèle calcule un score de risque de pluie à horizon d'une heure.

La réponse contient notamment :

```text
station
horizon de prévision
nombre d'observations utilisées
seuil de décision
nombre d'heures au-dessus du seuil
score maximal
score moyen
résultats heure par heure
```

Exemple réellement obtenu pour le `24/09/2026` :

```text
Observations utilisées : 24
Heures au-dessus du seuil : 3
Score maximal : 56.01 %
Score moyen : 21.70 %
Seuil de décision : 50 %
```

Le score produit par le modèle ne doit pas être interprété comme une probabilité météorologique calibrée.

---

## Interface Streamlit

L'interface utilisateur est développée dans :

```text
streamlit_app/app.py
```

Après avoir démarré FastAPI, ouvrir un second terminal et exécuter :

```powershell
python -m streamlit run streamlit_app/app.py
```

L'interface est alors accessible sur :

```text
http://localhost:8501
```

L'utilisateur peut sélectionner une date, vérifier la disponibilité du service, vérifier la présence du modèle puis lancer une estimation.

Lorsque des données exploitables existent pour la date demandée, Streamlit affiche :

```text
un message de synthèse
le score maximal
le score moyen
le nombre d'heures en alerte
le nombre d'observations utilisées
le seuil de décision
l'horizon de prévision
un graphique de l'évolution horaire
un tableau détaillé heure par heure
```

La date du `24/09/2026` est proposée par défaut afin de fournir une démonstration reproductible avec les données disponibles dans le MVP.

---

## Tests automatisés

Les tests sont exécutés avec :

```powershell
python -m pytest -q
```

Dernier résultat obtenu :

```text
20 passed, 1 warning
```

Les tests couvrent notamment le fonctionnement de l'API, le statut du modèle, les erreurs lorsque le modèle ou les données sont absents, la structure d'une prédiction valide, la validation des dates, la base SQLite, la validation du dataset, l'entraînement du modèle et la sauvegarde de l'artefact Machine Learning.

L'avertissement actuellement affiché concerne une dépréciation future de `TestClient` dans l'environnement FastAPI/Starlette. Il n'empêche pas l'exécution des tests ni le fonctionnement du MVP.

---

## Installation

Cloner le dépôt puis se placer dans son répertoire.

Créer l'environnement virtuel :

```powershell
python -m venv .venv
```

Sous Windows PowerShell, activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
pip install -r requirements.txt
```

---

## Configuration de la collecte Infoclimat

Les secrets ne sont jamais enregistrés dans Git.

Créer localement un fichier :

```text
.env
```

à partir du modèle :

```text
.env.example
```

La variable attendue est :

```text
INFOCLIMAT_API_KEY=
```

La clé réelle doit rester exclusivement dans le fichier `.env`.

Le fichier `.gitignore` empêche son ajout au dépôt.

Le modèle entraîné et le dataset préparé étant déjà inclus dans le dépôt, la clé Infoclimat n'est nécessaire que pour effectuer une nouvelle collecte de données.

---

## Versionnement des données et artefacts

Le dépôt Git contient :

```text
le code source
les tests
le dataset préparé weather_dataset.csv
le modèle final rain_model.joblib
la documentation
les fichiers de configuration non sensibles
```

Ne sont volontairement pas versionnés :

```text
le fichier .env
les clés API
les fichiers JSON bruts
la base SQLite locale
l'environnement virtuel
les caches Python et pytest
```

Cette organisation limite la taille du dépôt tout en conservant les éléments nécessaires à l'analyse et à la reproductibilité du modèle.

---

## Source et attribution des données

Les données météorologiques utilisées pour le périmètre pilote proviennent d'Infoclimat.

Station utilisée :

```text
000EW – Paris 20ème - Porte de Vincennes
```

La station est diffusée sous licence ouverte avec obligation de citer la source.

Attribution :

```text
Source des données météorologiques : Infoclimat – station 000EW,
Paris 20ème - Porte de Vincennes.
```

---

## Limites du MVP

Le prototype doit être interprété dans le périmètre pour lequel il a été développé.

Il repose actuellement sur une seule station pilote.

Le modèle est entraîné sur environ une année de données et la classe correspondant à la pluie reste minoritaire.

Le score retourné par la régression logistique n'a pas fait l'objet d'une calibration probabiliste.

La valeur `0.50` constitue un seuil de classification et non une garantie météorologique.

Le MVP réalise une estimation à horizon d'une heure à partir de variables météorologiques observées. Il ne constitue donc pas un système opérationnel de prévision météorologique à plusieurs jours.

La sélection d'une date dans l'interface correspond à un mode de démonstration basé sur les observations disponibles dans le périmètre du dataset.

Une utilisation opérationnelle en temps réel nécessiterait notamment l'intégration continue de nouvelles observations, une stratégie de réentraînement, une surveillance de la dérive du modèle et, pour des horizons plus longs, l'intégration de véritables données de prévision météorologique.

---

## Perspectives d'évolution

Les principales évolutions envisagées sont l'automatisation de l'alimentation des données, l'extension à plusieurs stations et régions, l'évaluation de modèles complémentaires, la calibration du score, la mise en place d'une validation temporelle plus complète, le suivi des performances dans le temps, l'ajout d'une stratégie de réentraînement, l'intégration éventuelle de données issues de capteurs IoT et le déploiement de l'API et de l'interface sur une infrastructure responsable.

---

## Qualité et reproductibilité

Le projet repose sur un environnement Python isolé, un fichier `requirements.txt`, un versionnement Git, des tests automatisés, une séparation claire entre les composants, une gestion distincte des informations sensibles, un dataset final versionné, un modèle entraîné versionné et une documentation des principales commandes nécessaires à l'exécution du MVP.

Ces choix facilitent la maintenance, l'audit du code et l'évolution progressive de la solution.