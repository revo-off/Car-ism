# API de Prédiction de Prix de Véhicules

API REST construite avec FastAPI pour prédire le prix des véhicules d'occasion en utilisant différents modèles de Machine Learning.

## Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances

```bash
cd api
pip install -r requirements.txt
```

## 🏃 Lancement de l'API

```bash
python main.py
```

L'API sera accessible sur `http://localhost:8000`

## Documentation

Une fois l'API lancée, accédez à la documentation interactive :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## Endpoints disponibles

### 1. Page d'accueil
```
GET /
```
Retourne les informations de base de l'API.

### 2. Liste des modèles
```
GET /models
```
Retourne la liste de tous les modèles disponibles avec leurs descriptions.

### 3. Prédiction avec un modèle spécifique
```
POST /predict/{model_name}
```
Prédit le prix avec le modèle spécifié.

**Modèles disponibles :**
- `knn` - K-Nearest Neighbors
- `random_forest` - Random Forest
- `linear_regression` - Régression Linéaire

**Exemple de requête :**
```json
{
  "brand": "toyota",
  "model": "corolla",
  "year": 2020,
  "mileage_km": 50000,
  "horsepower": 150,
  "condition_score": 8.5,
  "days_on_market": 15,
  "country": "japan",
  "fuel_type": "gasoline",
  "transmission": "automatic"
}
```

**Exemple de réponse :**
```json
{
  "model": "random_forest",
  "predicted_price": 18500.50,
  "input_features": { ... }
}
```

### 4. Prédiction avec tous les modèles
```
POST /predict/all
```
Prédit le prix avec tous les modèles disponibles simultanément.

**Exemple de réponse :**
```json
{
  "knn": 18200.00,
  "random_forest": 18500.50,
  "linear_regression": 17800.25
}
```

### 5. Health check
```
GET /health
```
Vérifie l'état de l'API et des modèles chargés.

## Modèles ML utilisés

Les modèles entraînés sont situés dans le dossier `../model/` :
- `knn_cars.pkl` - K-Nearest Neighbors
- `random_forest_cars.pkl` - Random Forest (300 arbres)
- `linear_regression_cars.pkl` - Régression Linéaire

## Notes

- Les modèles doivent être présents dans le dossier `model/` à la racine du projet
- L'API utilise le port 8000 par défaut
- CORS est activé pour permettre les requêtes depuis n'importe quelle origine