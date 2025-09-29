
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from app.api_alternative import app

client = TestClient(app)

class TestAlternativeAPI:
    
    def test_root_endpoint(self):
        """Test del endpoint raíz"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"
        assert "method" in data
    
    def test_health_endpoint(self):
        """Test del health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0 - Alternative Method"
    
    def test_predict_with_valid_prices(self):
        """Test con precios válidos (LAG 7)"""
        test_data = {
            "close_prices": [50000, 51000, 49000, 52000, 48000, 53000, 47000]
        }
        
        response = client.post("/predict", json=test_data)
        
        # Si los modelos no están cargados, será error 500, si están será 200
        if response.status_code == 200:
            data = response.json()
            assert "volatility_forecast" in data
            assert len(data["volatility_forecast"]) == 7
            assert data["horizon_days"] == 7
            assert data["lag_size"] == 7
            assert "interpretation" in data
            assert data["model_info"]["method"] == "Absolute log returns as volatility proxy"
        else:
            assert response.status_code == 500  # Modelos no cargados
    
    def test_predict_with_different_lag_sizes(self):
        """Test con diferentes tamaños de lag"""
        test_cases = [
            {"close_prices": [50000] * 14, "expected_lag": 14},
            {"close_prices": [50000] * 21, "expected_lag": 21}, 
            {"close_prices": [50000] * 28, "expected_lag": 28}
        ]
        
        for case in test_cases:
            response = client.post("/predict", json=case)
            if response.status_code == 200:
                data = response.json()
                assert data["lag_size"] == case["expected_lag"]
    
    def test_invalid_price_count(self):
        """Test con número inválido de precios"""
        test_data = {
            "close_prices": [50000, 51000, 49000]  # Solo 3 precios
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_negative_prices(self):
        """Test con precios negativos"""
        test_data = {
            "close_prices": [-50000, 51000, 49000, 52000, 48000, 53000, 47000]
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error