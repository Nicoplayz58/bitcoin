# Bitcoin Volatility Prediction API - Versión Alternativa

En esta variante se emplean los **valores absolutos de los retornos logarítmicos** como aproximación a la volatilidad realizada, en lugar de usar la desviación estándar móvil como medida de volatilidad histórica.

## Principales Cambios respecto a la Versión Base

### Enfoque Original:
- **Entrada**: Rezagos de volatilidad histórica (rolling std * √365)  
- **Salida**: Estimación de la volatilidad futura calculada  
- **Características**: Series de volatilidad preprocesada en ventanas móviles  

### Enfoque Alternativo:
- **Entrada**: Series de precios de cierre históricos  
- **Salida**: Valores futuros de |retornos logarítmicos| empleados como proxy de volatilidad  
- **Características**: Se utilizan precios en bruto, sin cálculo previo de volatilidad  

## Instalación y Ejecución

```bash
# 1. Entrenar los modelos de la versión alternativa
python train_and_save_alternative_models.py

# 2. Levantar la API en un puerto distinto
uvicorn app.api_alternative:app --reload --port 8001
