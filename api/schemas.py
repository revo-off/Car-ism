from pydantic import BaseModel, Field
from typing import Dict, Optional


class CarFeatures(BaseModel):
    brand: str = Field(..., description="Marque du véhicule", example="toyota")
    model: str = Field(..., description="Modèle du véhicule", example="corolla")
    year: int = Field(..., ge=1900, le=2026, description="Année de fabrication", example=2020)
    mileage_km: float = Field(..., ge=0, description="Kilométrage en km", example=50000)
    horsepower: float = Field(..., ge=0, description="Puissance en chevaux", example=150)
    condition_score: float = Field(..., ge=0, le=10, description="Score de condition (0-10)", example=8.5)
    days_on_market: int = Field(..., ge=0, description="Jours sur le marché", example=15)
    country: str = Field(..., description="Pays d'origine", example="japan")
    fuel_type: str = Field(..., description="Type de carburant", example="gasoline")
    transmission: str = Field(..., description="Type de transmission", example="automatic")

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    model: str = Field(..., description="Nom du modèle utilisé")
    predicted_price: float = Field(..., description="Prix prédit en USD")
    input_features: Dict = Field(..., description="Features utilisées pour la prédiction")

    class Config:
        schema_extra = {
            "example": {
                "model": "random_forest",
                "predicted_price": 18500.50,
                "input_features": {
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
            }
        }


class ModelInfo(BaseModel):
    name: str = Field(..., description="Nom du modèle")
    description: str = Field(..., description="Description du modèle")
    available: bool = Field(..., description="Disponibilité du modèle")

    class Config:
        schema_extra = {
            "example": {
                "name": "random_forest",
                "description": "Random Forest - Modèle d'ensemble avec 300 arbres de décision",
                "available": True
            }
        }
