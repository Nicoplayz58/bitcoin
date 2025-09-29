
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# BDS: intentamos statsmodels; si no, arch
_BDS_BACKEND = None
try:
    from statsmodels.tsa.stattools import bds as bds_sm
    _BDS_BACKEND = "statsmodels"
except Exception:
    try:
        from arch.bootstrap import bds as bds_arch
        _BDS_BACKEND = "arch"
    except Exception:
        _BDS_BACKEND = None

# Intentar usar tsxv/timeseries-cv (como exige el enunciado)
_TSX_AVAILABLE = False
try:
    from tsxv import split_train_val_test_groupKFold as _split_tsxv
    _TSX_AVAILABLE = True
except Exception:
    try:
        from timeseries_cv import split_train_val_test_groupKFold as _split_tsxv
        _TSX_AVAILABLE = True
    except Exception:
        _TSX_AVAILABLE = False

# Rutas de salida
FIGS_DIR = Path("notebooks/figs")
RESULTS_DIR = Path("results")
FIGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Path('app/models').mkdir(parents=True, exist_ok=True)

print("ENTRENAMIENTO EXACTO BASADO EN BITCOIN_EDA.PY")
print("=" * 60)

# CARGAR DATOS EXACTO COMO EN EL ORIGINAL
btc = pd.read_csv('btc_1d_data_2018_to_2025.csv')
btc = btc[['Close time', 'Close']].dropna().reset_index(drop=True)
btc = btc.rename(columns={"Close time": "Date"})
btc["Date"] = pd.to_datetime(btc["Date"])
btc = btc.sort_values("Date").reset_index(drop=True)

# Retornos log
btc["log_return"] = np.log(btc["Close"] / btc["Close"].shift(1))

# Limpia NaN inicial
btc = btc.dropna().reset_index(drop=True)

dates_full = btc["Date"].copy()
close_full = btc["Close"].values.astype(float)
ret_full   = btc["log_return"].values.astype(float)

print(f"Datos cargados: {len(btc)} observaciones")
print(f"Rango temporal: {btc['Date'].min()} a {btc['Date'].max()}")

def build_supervised_from_close_for_volatility(close: np.ndarray,
                                               log_returns: np.ndarray,
                                               dates: pd.Series,
                                               n_steps_input: int,
                                               n_steps_forecast: int,
                                               n_steps_jump: int = 1):
    """
    EXACTA del código original bitcoin_eda.py
    Features X: últimos 'n_steps_input' CIERRES (Close).
    Target Y:  |retornos log| de los próximos 'n_steps_forecast' días (proxy de volatilidad realizada).
    """
    assert len(close) == len(log_returns) == len(dates)
    X_list, Y_list, D_list = [], [], []
    T = len(close)

    for t in range(n_steps_input - 1, T - n_steps_forecast - 1, n_steps_jump):
        # ventana de input (closes)
        x = close[t - n_steps_input + 1 : t + 1]
        # futuros retornos (abs para proxy de volatilidad realizada)
        future_r = np.abs(log_returns[t + 1 : t + 1 + n_steps_forecast])
        if len(x) == n_steps_input and len(future_r) == n_steps_forecast:
            X_list.append(x)
            Y_list.append(future_r)
            D_list.append(dates.iloc[t])

    X = np.asarray(X_list)
    Y = np.asarray(Y_list)
    D = pd.to_datetime(pd.Series(D_list))
    return X, Y, D

def make_time_groups(dates: pd.Series, freq: str = "W"):
    """EXACTA del código original"""
    period = pd.to_datetime(dates).dt.to_period(freq).astype(str)
    return pd.Categorical(period).codes

def _fallback_split_train_val_test_groupKFold(X, Y, dates, n_splits=5, val_size=0.5, freq_groups="W"):
    """EXACTA del código original"""
    groups = make_time_groups(dates, freq=freq_groups)
    gkf = GroupKFold(n_splits=n_splits)

    splits = []
    for tr_idx, ho_idx in gkf.split(X, Y, groups=groups):
        tr_idx = np.sort(tr_idx)
        ho_idx = np.sort(ho_idx)

        order = np.argsort(dates.iloc[ho_idx].values)
        ho_ord = ho_idx[order]
        n_hold = len(ho_ord)
        n_val = max(1, min(int(np.floor(val_size * n_hold)), n_hold - 1))

        va_idx = ho_ord[:n_val]
        te_idx = ho_ord[n_val:]

        splits.append({"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx})
    return splits

def split_train_val_test_groupKFold_wrapper(X, Y, dates, n_splits=5, val_size=0.5, freq_groups="W"):
    """EXACTA del código original"""
    if _TSX_AVAILABLE:
        try:
            groups = make_time_groups(dates, freq=freq_groups)
            return _split_tsxv(X=X, y=Y, groups=groups, n_splits=n_splits, val_size=val_size, shuffle=False)
        except TypeError:
            return _fallback_split_train_val_test_groupKFold(X, Y, dates, n_splits, val_size, freq_groups)
    else:
        return _fallback_split_train_val_test_groupKFold(X, Y, dates, n_splits, val_size, freq_groups)

def safe_mape(y_true, y_pred, eps=1e-8):
    """EXACTA del código original"""
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom), axis=0)

def mae(y_true, y_pred):
    """EXACTA del código original"""
    return np.mean(np.abs(y_true - y_pred), axis=0)

def mse(y_true, y_pred):
    """EXACTA del código original"""
    return np.mean((y_true - y_pred)**2, axis=0)

def rmse(y_true, y_pred):
    """EXACTA del código original"""
    return np.sqrt(mse(y_true, y_pred))

def get_bds_pvalue(residuals_h1, max_dim=2, min_len=40):
    """EXACTA del código original"""
    r = np.asarray(residuals_h1).astype(float)
    r = r[np.isfinite(r)]

    if r.size < min_len:
        return np.nan

    mu, sd = np.mean(r), np.std(r, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    z = (r - mu) / sd

    if _BDS_BACKEND == "statsmodels":
        try:
            res = bds_sm(z, max_dim=max_dim)
            return float(getattr(res, "pvalue", np.nan))
        except Exception:
            return np.nan
    elif _BDS_BACKEND == "arch":
        try:
            res = bds_arch(z, max_dim=max_dim)
            pv_col = None
            for c in res.columns:
                if "p" in c.lower():
                    pv_col = c; break
            if pv_col is None:
                return np.nan
            if 2 in getattr(res, "index", []):
                return float(res.loc[2, pv_col])
            else:
                return float(res.iloc[1][pv_col])
        except Exception:
            return np.nan
    else:
        return np.nan

# CONFIGURACIÓN EXACTA DEL ORIGINAL
lags_list = [7, 14, 21, 28]
n_steps_forecast = 7
n_steps_jump = 1

results_by_L = {}
metrics_rows_master = []

# SAVE_ARTIFACTS = True (siempre guardar para API)
best_models = {}  # Para guardar el mejor modelo por lag

print("\n=== COMENZANDO ENTRENAMIENTO EXACTO ===")

for L in lags_list:
    print(f"\n===== Usando {L} lags del precio → predecir |ret| futuros (h=1..7) =====")

    # 1) Dataset supervisado EXACTO
    X, Y, D = build_supervised_from_close_for_volatility(
        close=close_full, log_returns=ret_full, dates=dates_full,
        n_steps_input=L, n_steps_forecast=n_steps_forecast, n_steps_jump=n_steps_jump
    )
    print(f"Shapes -> X: {X.shape}, Y: {Y.shape}, D: {len(D)}")

    # 2) Splits temporales EXACTOS
    splits = split_train_val_test_groupKFold_wrapper(
        X=X, Y=Y, dates=D, n_splits=5, val_size=0.5, freq_groups="W"
    )
    print(f"Folds: {len(splits)}")

    fold_metrics = []
    rmse_folds = []
    bds_pvals = []
    y_test_list = []
    y_pred_list = []
    fold_models = []  # Para guardar modelos de cada fold

    for k, sp in enumerate(splits, start=1):
        tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]
        X_tr, X_va, X_te = X[tr], X[va], X[te]
        Y_tr, Y_va, Y_te = Y[tr], Y[va], Y[te]

        # Escalado sin leakage EXACTO
        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(Y_tr)
        X_tr_s, X_va_s, X_te_s = sx.transform(X_tr), sx.transform(X_va), sx.transform(X_te)
        Y_tr_s, Y_va_s, Y_te_s = sy.transform(Y_tr), sy.transform(Y_va), sy.transform(Y_te)

        # Modelo EXACTO COMO EL ORIGINAL
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                             max_iter=300, random_state=42 + k)
        model.fit(X_tr_s, Y_tr_s)

        # Predicciones TEST (des-escala) EXACTO
        Yhat_te_s = model.predict(X_te_s)
        Yhat_te = sy.inverse_transform(Yhat_te_s)
        Y_te_real = Y_te

        # Métricas por horizonte y promedios EXACTO
        mape_h = safe_mape(Y_te_real, Yhat_te)
        mae_h  = mae(Y_te_real, Yhat_te)
        mse_h  = mse(Y_te_real, Yhat_te)
        rmse_h = rmse(Y_te_real, Yhat_te)

        mape_avg = float(np.mean(mape_h))
        mae_avg  = float(np.mean(mae_h))
        mse_avg  = float(np.mean(mse_h))
        rmse_avg = float(np.mean(rmse_h))

        # BDS en h1 EXACTO
        resid_h1 = Y_te_real[:, 0] - Yhat_te[:, 0]
        bds_p = float(get_bds_pvalue(resid_h1))

        fold_row = {
            "L": L, "fold": k,
            **{f"MAPE_h{h+1}": mape_h[h] for h in range(n_steps_forecast)},
            **{f"MAE_h{h+1}":  mae_h[h]  for h in range(n_steps_forecast)},
            **{f"MSE_h{h+1}":  mse_h[h]  for h in range(n_steps_forecast)},
            **{f"RMSE_h{h+1}": rmse_h[h] for h in range(n_steps_forecast)},
            "MAPE_avg": mape_avg, "MAE_avg": mae_avg, "MSE_avg": mse_avg, "RMSE_avg": rmse_avg,
            "BDS_pvalue_h1": bds_p
        }
        fold_metrics.append(fold_row)
        rmse_folds.append(rmse_avg)
        bds_pvals.append(bds_p)

        y_test_list.append(Y_te_real)
        y_pred_list.append(Yhat_te)
        
        # Guardar modelo y scalers de este fold
        fold_models.append({
            'model': model,
            'scaler_x': sx,
            'scaler_y': sy,
            'rmse': rmse_avg,
            'fold': k
        })

    # DataFrame por L EXACTO
    df_metrics = pd.DataFrame(fold_metrics)

    # Guardar CSV por L
    df_metrics.to_csv(RESULTS_DIR / f"alt_metrics_test_L{L}.csv", index=False)

    # Almacenar en memoria EXACTO
    results_by_L[L] = {
        "df_metrics": df_metrics,
        "rmse_folds": rmse_folds,
        "bds_pvals": bds_pvals,
        "y_test_list": y_test_list,
        "y_pred_list": y_pred_list
    }

    # SELECCIONAR MEJOR MODELO DEL FOLD CON MENOR RMSE
    best_fold_idx = np.argmin(rmse_folds)
    best_model_info = fold_models[best_fold_idx]
    
    print(f"Mejor fold: {best_model_info['fold']} con RMSE: {best_model_info['rmse']:.6f}")
    
    # ENTRENAR MODELO FINAL CON TODOS LOS DATOS (como en método original de producción)
    final_sx = StandardScaler().fit(X)
    final_sy = StandardScaler().fit(Y)
    X_final_s = final_sx.transform(X)
    Y_final_s = final_sy.transform(Y)
    
    final_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                               max_iter=300, random_state=42)
    final_model.fit(X_final_s, Y_final_s)
    
    # Guardar modelo final y scalers
    import joblib
    joblib.dump(final_model, f'app/models/alt_model_L{L}.pkl')
    joblib.dump(final_sx, f'app/models/alt_scaler_x_L{L}.pkl')
    joblib.dump(final_sy, f'app/models/alt_scaler_y_L{L}.pkl')
    
    best_models[L] = {
        'model': final_model,
        'scaler_x': final_sx,
        'scaler_y': final_sy,
        'metrics': {
            'rmse_mean': np.mean(rmse_folds),
            'rmse_std': np.std(rmse_folds),
            'mae_mean': df_metrics["MAE_avg"].mean(),
            'mape_mean': df_metrics["MAPE_avg"].mean(),
            'bds_mean': df_metrics["BDS_pvalue_h1"].mean()
        }
    }
    
    print(f"✓ Modelo final guardado: app/models/alt_model_L{L}.pkl")

    # Acumular resumen para tabla maestra EXACTO
    metrics_rows_master.append({
        "L": L,
        "MAPE_avg_mean": df_metrics["MAPE_avg"].mean(), "MAPE_avg_std": df_metrics["MAPE_avg"].std(),
        "MAE_avg_mean":  df_metrics["MAE_avg"].mean(),  "MAE_avg_std":  df_metrics["MAE_avg"].std(),
        "MSE_avg_mean":  df_metrics["MSE_avg"].mean(),  "MSE_avg_std":  df_metrics["MSE_avg"].std(),
        "RMSE_avg_mean": df_metrics["RMSE_avg"].mean(), "RMSE_avg_std": df_metrics["RMSE_avg"].std(),
        "BDS_pvalue_h1_mean": df_metrics["BDS_pvalue_h1"].mean()
    })

print("✓ Sección 5 completada EXACTA como bitcoin_eda.py")

# TABLA MAESTRA EXACTA
df_master = pd.DataFrame(metrics_rows_master).sort_values("L").reset_index(drop=True)

def fmt(m, s):
    return f"{m:.6f} ± {s:.6f}"

tabla_maestra = pd.DataFrame({
    "L": df_master["L"],
    "MAPE_avg (mean±std)": [fmt(m, s) for m, s in zip(df_master["MAPE_avg_mean"], df_master["MAPE_avg_std"])],
    "MAE_avg  (mean±std)": [fmt(m, s) for m, s in zip(df_master["MAE_avg_mean"],  df_master["MAE_avg_std"])],
    "MSE_avg  (mean±std)": [fmt(m, s) for m, s in zip(df_master["MSE_avg_mean"],  df_master["MSE_avg_std"])],
    "RMSE_avg (mean±std)": [fmt(m, s) for m, s in zip(df_master["RMSE_avg_mean"], df_master["RMSE_avg_std"])],
    "BDS_pvalue_h1_mean":  [f"{v:.6f}" for v in df_master["BDS_pvalue_h1_mean"]]
})

print("\n" + "="*80)
print("TABLA MAESTRA FINAL (EXACTA COMO BITCOIN_EDA.PY)")
print("="*80)
print(tabla_maestra.to_string(index=False))

# Guardar tablas
df_master.to_csv(RESULTS_DIR / "alt_summary_by_L_raw.csv", index=False)
tabla_maestra.to_csv(RESULTS_DIR / "alt_summary_by_L_formatted.csv", index=False)

print("\n✓ ENTRENAMIENTO COMPLETADO EXACTO COMO BITCOIN_EDA.PY")
print("✓ Modelos guardados en app/models/alt_*")
print("✓ Tablas guardadas en results/alt_*")
print("\nPara probar la API:")
print("uvicorn app.api_alternative:app --reload --port 8001")
