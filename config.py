"""
Configuración para el Modelo Avanzado de Forecasting
===================================================
Define parámetros, rutas y configuraciones globales.
"""

import os
from datetime import datetime

# Rutas principales
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Crear directorios si no existen
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Parámetros del modelo
RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'n_ventas'
FORECAST_HORIZON = 12  # Número de meses a predecir
CV_FOLDS = 5  # Número de folds para validación cruzada temporal
OPTUNA_TRIALS = 200  # Número de pruebas para optimización

# Parámetros avanzados
OUTLIER_THRESHOLD = 3.5  # Umbral para detección de outliers (usando MAD)
ENSEMBLE_MODELS = 5  # Número de modelos en el ensemble
BOOSTING_ROUNDS = 10000  # Máximo número de rondas para early stopping
EARLY_STOPPING = 50  # Paciencia para early stopping
FEATURE_SELECTION_THRESHOLD = 0.01  # Umbral mínimo para importancia de características

# Parámetros para transformaciones
TRANSFORMATION_METHODS = ['none', 'log', 'sqrt', 'boxcox', 'yeojohnson']

# Archivo de datos
DATA_FILE = os.path.join(DATA_DIR, "serie_temporal_mensual_cluster1.csv")

# Nombre del modelo final
MODEL_NAME = f"xgboost_ultimate_{datetime.now().strftime('%Y%m%d_%H%M')}"

# Configuración para logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOGS_DIR, f"model_training_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
