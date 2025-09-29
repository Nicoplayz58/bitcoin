
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

def build_supervised_from_close_for_volatility(close: np.ndarray,
                                               log_returns: np.ndarray, 
                                               dates: pd.Series,
                                               n_steps_input: int,
                                               n_steps_forecast: int,
                                               n_steps_jump: int = 1):
    """
    Construye dataset supervisado:
    - X: precios de cierre históricos (n_steps_input)
    - Y: |retornos logarítmicos| futuros (n_steps_forecast) como proxy volatilidad
    """
    assert len(close) == len(log_returns) == len(dates)
    X_list, Y_list, D_list = [], [], []
    T = len(close)
    
    for t in range(n_steps_input - 1, T - n_steps_forecast - 1, n_steps_jump):
        # Features: precios de cierre históricos
        x = close[t - n_steps_input + 1 : t + 1]
        # Target: |retornos| futuros como proxy de volatilidad
        future_r = np.abs(log_returns[t + 1 : t + 1 + n_steps_forecast])
        
        if len(x) == n_steps_input and len(future_r) == n_steps_forecast:
            X_list.append(x)
            Y_list.append(future_r)
            D_list.append(dates.iloc[t])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    D = pd.Series(D_list)
    
    return X, Y, D

def safe_mape(y_true, y_pred, eps=1e-8):
    """MAPE seguro para evitar división por cero"""
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom), axis=0)

def mae_per_horizon(y_true, y_pred):
    """MAE por horizonte"""
    return np.mean(np.abs(y_true - y_pred), axis=0)

def mse_per_horizon(y_true, y_pred):
    """MSE por horizonte"""
    return np.mean((y_true - y_pred)**2, axis=0)

def rmse_per_horizon(y_true, y_pred):
    """RMSE por horizonte"""
    return np.sqrt(mse_per_horizon(y_true, y_pred))

def save_alternative_model(model, scaler_x, scaler_y, lag_size, metrics=None):
    """Guardar artifacts del modelo alternativo"""
    Path('app/models').mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, f'app/models/alt_model_L{lag_size}.pkl')
    joblib.dump(scaler_x, f'app/models/alt_scaler_x_L{lag_size}.pkl')
    joblib.dump(scaler_y, f'app/models/alt_scaler_y_L{lag_size}.pkl')
    
    if metrics:
        pd.DataFrame([metrics]).to_csv(f'app/models/alt_metrics_L{lag_size}.csv', index=False)

def load_and_prepare_btc_data(filepath='btc_1d_data_2018_to_2025.csv'):
    """
    Cargar y preparar datos de Bitcoin para la versión alternativa
    """
    btc = pd.read_csv(filepath)
    btc = btc[['Close time', 'Close']].dropna().reset_index(drop=True)
    btc = btc.rename(columns={"Close time": "Date"})
    btc["Date"] = pd.to_datetime(btc["Date"])
    btc = btc.sort_values("Date").reset_index(drop=True)
    
    # Retornos logarítmicos  
    btc["log_return"] = np.log(btc["Close"] / btc["Close"].shift(1))
    btc = btc.dropna().reset_index(drop=True)
    
    return btc