from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List
import uvicorn

from schemas import CarFeatures, PredictionResponse, ModelInfo

app = FastAPI(
    title="Car-ism",
    description="API pour prédire le prix des véhicules d'occasion avec différents modèles ML",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles
MODEL_DIR = Path(__file__).parent.parent / "model"
MODELS = {}

def load_models():
    model_files = {
        "knn": "knn_cars.pkl",
        "random_forest": "random_forest_cars.pkl",
        "linear_regression": "linear_regression_cars.pkl"
    }
    
    for name, file in model_files.items():
        model_path = MODEL_DIR / file
        if model_path.exists():
            try:
                MODELS[name] = joblib.load(model_path)
            except Exception as e:
                print(f"Erreur  de chargement de {name}: {e}")
        else:
            print(f"Fichier {file} introuvable")

# Charger les modèles au démarrage
load_models()


@app.on_event("startup")
async def startup_event():
    print("="*50)
    print("Car-ism API démarrée")
    print(f"Modèles disponibles: {list(MODELS.keys())}")
    print("="*50)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Bienvenue sur l'API de Car-ism !",
        "docs": "/docs",
        "models_available": list(MODELS.keys())
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def get_models():
    return [
        ModelInfo(
            name=name,
            description=get_model_description(name),
            available=True
        )
        for name in MODELS.keys()
    ]


def get_model_description(model_name: str) -> str:
    descriptions = {
        "knn": "K-Nearest Neighbors - Prédit le prix basé sur les véhicules similaires",
        "random_forest": "Random Forest - Modèle d'ensemble avec 300 arbres de décision",
        "linear_regression": "Régression Linéaire - Modèle baseline simple et interprétable"
    }
    return descriptions.get(model_name, "Aucune description disponible")


@app.post("/predict/{model_name}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_price(model_name: str, car: CarFeatures):
    if model_name not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' non trouvé. Modèles disponibles: {list(MODELS.keys())}"
        )
    
    try:
        # Convertir les features en DataFrame
        car_dict = car.dict()
        df = pd.DataFrame([car_dict])
        
        # Prédiction
        model = MODELS[model_name]
        prediction = model.predict(df)[0]
        
        return PredictionResponse(
            model=model_name,
            predicted_price=round(float(prediction), 2),
            input_features=car_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post("/predict/all", response_model=Dict[str, float], tags=["Predictions"])
async def predict_all_models(car: CarFeatures):
    try:
        car_dict = car.dict()
        df = pd.DataFrame([car_dict])
        
        predictions = {}
        for model_name, model in MODELS.items():
            try:
                pred = model.predict(df)[0]
                predictions[model_name] = round(float(pred), 2)
            except Exception as e:
                predictions[model_name] = f"Erreur: {str(e)}"
        
        return predictions
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "models": list(MODELS.keys())
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
