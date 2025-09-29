
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

app = FastAPI(
    title="Bitcoin Volatility Prediction API - Alternative Version",
    description="API para predecir volatilidad usando |retornos| y precios históricos de cierre",
    version="2.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos y scalers cargados globalmente
models = {}
scalers = {}

def load_models():
    """Cargar modelos entrenados y scalers"""
    try:
        lag_sizes = [7, 14, 21, 28]
        for lag in lag_sizes:
            models[lag] = joblib.load(f'app/models/alt_model_L{lag}.pkl')
            scalers[lag] = {
                'scaler_x': joblib.load(f'app/models/alt_scaler_x_L{lag}.pkl'),
                'scaler_y': joblib.load(f'app/models/alt_scaler_y_L{lag}.pkl')
            }
        logger.info("Modelos alternativos cargados exitosamente")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_models()

# Modelos Pydantic
class PredictionRequest(BaseModel):
    close_prices: List[float]
    
    @field_validator('close_prices')
    @classmethod
    def validate_prices(cls, v):
        if len(v) not in [7, 14, 21, 28]:
            raise ValueError('close_prices debe tener longitud 7, 14, 21 o 28')
        if any(x <= 0 for x in v):
            raise ValueError('Todos los precios deben ser positivos')
        return v

class PredictionResponse(BaseModel):
    volatility_forecast: List[float]
    horizon_days: int
    lag_size: int
    model_info: dict
    interpretation: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[int]
    version: str

# Endpoints
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Bitcoin Volatility Prediction API - Alternative Version",
        "version": "2.0.0",
        "method": "Uses |log_returns| as realized volatility proxy",
        "input": "Historical close prices",
        "output": "7-day |return| forecasts",
        "endpoints": {
            "/predict": "POST - Predecir volatilidad usando precios históricos",
            "/health": "GET - Estado de la API",
            "/docs": "GET - Documentación automática"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        version="2.0.0 - Alternative Method"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_volatility(request: PredictionRequest):
    """
    Predecir volatilidad usando precios de cierre históricos
    
    Args:
        request: Precios de cierre históricos (7, 14, 21 o 28 días)
        
    Returns:
        Predicción de |retornos| para próximos 7 días (proxy de volatilidad)
    """
    try:
        prices = np.array(request.close_prices)
        lag_size = len(prices)
        
        if lag_size not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Modelo no disponible para lag_size={lag_size}"
            )
        
        # Preparar datos (usar precios directamente como features)
        X = prices.reshape(1, -1)
        
        # Escalar
        scaler_x = scalers[lag_size]['scaler_x']
        scaler_y = scalers[lag_size]['scaler_y']
        X_scaled = scaler_x.transform(X)
        
        # Predecir
        model = models[lag_size]
        y_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)
        
        # Formatear respuesta
        forecast = y_pred[0].tolist()
        
        # Interpretación de resultados
        avg_volatility = np.mean(forecast)
        max_volatility = np.max(forecast)
        
        if avg_volatility < 0.02:
            risk_level = "BAJO"
        elif avg_volatility < 0.05:
            risk_level = "MODERADO"
        else:
            risk_level = "ALTO"
            
        interpretation = (
            f"Volatilidad promedio predicha: {avg_volatility:.4f} ({avg_volatility*100:.2f}%). "
            f"Pico máximo: {max_volatility:.4f} ({max_volatility*100:.2f}%). "
            f"Nivel de riesgo: {risk_level}."
        )
        
        return PredictionResponse(
            volatility_forecast=forecast,
            horizon_days=7,
            lag_size=lag_size,
            model_info={
                "model_type": "MLP",
                "architecture": str(getattr(model, 'hidden_layer_sizes', 'N/A')),
                "activation": getattr(model, 'activation', 'N/A'),
                "method": "Absolute log returns as volatility proxy",
                "input_type": "Historical close prices"
            },
            interpretation=interpretation
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))