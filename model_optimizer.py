"""
Optimización Avanzada de Modelos para Forecasting
=================================================
Implementa técnicas avanzadas de optimización de hiperparámetros
para modelos de forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import os
import pickle
import logging
import time
from datetime import datetime
from config import *

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "model_optimizer.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelOptimizer")

class ModelOptimizer:
    """Clase para optimización avanzada de modelos de forecasting."""
    
    def __init__(self, X_train, y_train, X_test=None, y_test=None, 
                 cv_folds=CV_FOLDS, n_trials=OPTUNA_TRIALS, random_state=RANDOM_SEED):
        """
        Inicializa el optimizador de modelos.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_test: Características de prueba (opcional)
            y_test: Target de prueba (opcional)
            cv_folds: Número de folds para validación cruzada
            n_trials: Número de pruebas para Optuna
            random_state: Semilla aleatoria
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.optimization_history = None
        self.feature_importances = None
        self.study = None
        
    def time_series_cv(self, model, X, y, n_splits=None):
        """
        Realiza validación cruzada temporal.
        
        Args:
            model: Modelo a evaluar
            X: Características
            y: Target
            n_splits: Número de divisiones (si None, usa self.cv_folds)
            
        Returns:
            Métrica promedio (MAE)
        """
        if n_splits is None:
            n_splits = self.cv_folds
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            
            y_pred = model.predict(X_val_cv)
            mae = mean_absolute_error(y_val_cv, y_pred)
            cv_scores.append(mae)
            
        return np.mean(cv_scores)
        
    def optimize_xgboost(self):
        """
        Optimiza hiperparámetros para XGBoost usando Optuna.
        
        Returns:
            dict: Mejores hiperparámetros encontrados
        """
        logger.info(f"Iniciando optimización de XGBoost con {self.n_trials} trials")
        start_time = time.time()
        
        def objective(trial):
            # Espacio de hiperparámetros - muy amplio para encontrar configuración óptima
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # Crear modelo con los parámetros actuales
            model = XGBRegressor(**params)
            
            # Evaluar usando validación cruzada temporal
            cv_score = self.time_series_cv(model, self.X_train, self.y_train)
            
            # Para mostrar progreso
            if trial.number % 10 == 0:
                logger.info(f"Trial {trial.number}/{self.n_trials}: MAE = {cv_score:.4f}")
                
            return cv_score
        
        # Crear estudio Optuna con sampler TPE (más eficiente)
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Ejecutar optimización
        study.optimize(objective, n_trials=self.n_trials)
        
        # Guardar resultado
        best_params = study.best_params
        best_value = study.best_value
        
        # Agregar parámetros fijos
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        
        # Duración
        duration = time.time() - start_time
        
        logger.info(f"Optimización completada en {duration:.1f} segundos")
        logger.info(f"Mejor MAE CV: {best_value:.4f}")
        logger.info(f"Mejores parámetros: {best_params}")
        
        # Guardar historia de optimización
        self.best_params = best_params
        self.study = study
        
        # Graficar historia de optimización
        self.plot_optimization_history()
        
        # Graficar importancia de hiperparámetros
        self.plot_param_importances()
        
        return best_params
    
    def train_best_model(self):
        """
        Entrena el modelo con los mejores hiperparámetros.
        
        Returns:
            Modelo entrenado
        """
        if self.best_params is None:
            logger.warning("No hay parámetros optimizados. Ejecutando optimización...")
            self.optimize_xgboost()
            
        logger.info("Entrenando modelo con los mejores hiperparámetros")
        
        model = XGBRegressor(**self.best_params)
        model.fit(self.X_train, self.y_train)
        
        # Guardar modelo
        self.best_model = model
        
        # Calcular importancia de características
        self.feature_importances = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Graficar importancia de características
        self.plot_feature_importances()
        
        # Evaluar si hay datos de test
        if self.X_test is not None and self.y_test is not None:
            self.evaluate_model()
            
        return model
    
    def evaluate_model(self):
        """
        Evalúa el modelo en el conjunto de prueba.
        
        Returns:
            dict: Métricas de evaluación
        """
        if self.best_model is None:
            logger.warning("No hay modelo entrenado. Entrenando primero...")
            self.train_best_model()
            
        if self.X_test is None or self.y_test is None:
            logger.warning("No hay datos de prueba para evaluación")
            return None
            
        logger.info("Evaluando modelo en datos de prueba")
        
        # Predecir
        y_pred = self.best_model.predict(self.X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        
        # Métricas adicionales
        # Error porcentual medio (MPE) - puede indicar sesgo
        mpe = np.mean((self.y_test - y_pred) / self.y_test) * 100
        
        # Coeficiente de determinación (R²)
        ss_total = np.sum((self.y_test - np.mean(self.y_test))**2)
        ss_residual = np.sum((self.y_test - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        
        # Mostrar resultados
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAPE: {mape:.4f}%")
        logger.info(f"MPE: {mpe:.4f}% (un valor cercano a 0 indica ausencia de sesgo)")
        logger.info(f"R²: {r2:.4f}")
        
        # Guardar métricas
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mpe': mpe,
            'r2': r2
        }
        
        # Guardar en archivo
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"), index=False)
        
        # Graficar predicciones vs reales
        self.plot_predictions(y_pred)
        
        return metrics
    
    def plot_optimization_history(self):
        """Grafica la historia de optimización."""
        if self.study is None:
            return
            
        # Extraer historia
        history = [trial.value for trial in self.study.trials]
        best_history = np.minimum.accumulate(history)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history, 'o-', alpha=0.3, label='Trials')
        plt.plot(best_history, 'r-', label='Mejor valor')
        plt.xlabel('Número de trial')
        plt.ylabel('MAE CV')
        plt.title('Historia de Optimización')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(history, bins=30, alpha=0.5)
        plt.axvline(x=self.study.best_value, color='r', linestyle='--', 
                  label=f'Mejor valor: {self.study.best_value:.4f}')
        plt.xlabel('MAE CV')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Resultados')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "optimization_history.png"))
        plt.close()
    
    def plot_param_importances(self):
        """Grafica la importancia de los hiperparámetros."""
        if self.study is None:
            return
            
        # Obtener importancias
        try:
            param_importances = optuna.importance.get_param_importances(self.study)
            
            # Convertir a DataFrame
            importance_df = pd.DataFrame({
                'parameter': list(param_importances.keys()),
                'importance': list(param_importances.values())
            }).sort_values('importance', ascending=False)
            
            # Graficar
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='parameter', data=importance_df)
            plt.title('Importancia de Hiperparámetros')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "param_importances.png"))
            plt.close()
            
            # Guardar en archivo
            importance_df.to_csv(os.path.join(RESULTS_DIR, "param_importances.csv"), index=False)
        except Exception as e:
            logger.warning(f"Error al calcular importancia de hiperparámetros: {e}")
    
    def plot_feature_importances(self, top_n=30):
        """Grafica la importancia de las características."""
        if self.feature_importances is None:
            return
            
        # Limitar a top_n
        n_features = min(top_n, len(self.feature_importances))
        top_features = self.feature_importances.head(n_features)
        
        plt.figure(figsize=(10, n_features * 0.3))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Importancia de Características')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importances.png"))
        plt.close()
        
        # Guardar en archivo
        self.feature_importances.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)
    
    def plot_predictions(self, y_pred):
        """
        Grafica predicciones vs valores reales.
        
        Args:
            y_pred: Predicciones
        """
        if self.y_test is None:
            return
            
        # Crear DataFrame con fechas
        df_plot = pd.DataFrame({
            'y_real': self.y_test,
            'y_pred': y_pred
        })
        
        # Calcular error
        df_plot['error'] = df_plot['y_real'] - df_plot['y_pred']
        df_plot['error_abs'] = np.abs(df_plot['error'])
        df_plot['error_pct'] = np.abs(df_plot['error'] / df_plot['y_real']) * 100
        
        # Gráfico principal
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df_plot.index, df_plot['y_real'], 'b-', label='Real')
        plt.plot(df_plot.index, df_plot['y_pred'], 'r--', label='Predicción')
        plt.title('Predicciones vs Valores Reales')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.bar(df_plot.index, df_plot['error'], color='g', alpha=0.6, label='Error')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Error de Predicción')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "predictions.png"))
        plt.close()
        
        # Scatter plot de real vs predicción
        plt.figure(figsize=(8, 8))
        plt.scatter(df_plot['y_real'], df_plot['y_pred'], alpha=0.6)
        
        # Línea de identidad (predicción perfecta)
        min_val = min(df_plot['y_real'].min(), df_plot['y_pred'].min())
        max_val = max(df_plot['y_real'].max(), df_plot['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.xlabel('Valor Real')
        plt.ylabel('Predicción')
        plt.title('Predicción vs Real')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "prediction_scatter.png"))
        plt.close()
        
        # Guardar valores en archivo
        df_plot.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"))
        
    def save_model(self, filename=None):
        """
        Guarda el modelo optimizado.
        
        Args:
            filename: Nombre del archivo (si None, usa timestamp)
            
        Returns:
            Ruta al archivo guardado
        """
        if self.best_model is None:
            logger.warning("No hay modelo entrenado para guardar")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xgboost_optimized_{timestamp}.pkl"
            
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Guardar modelo
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
            
        logger.info(f"Modelo guardado en: {filepath}")
        
        # Guardar también los parámetros
        params_file = os.path.splitext(filepath)[0] + "_params.json"
        pd.Series(self.best_params).to_json(params_file)
        
        return filepath
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath: Ruta al archivo del modelo
            
        Returns:
            Modelo cargado
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                
            self.best_model = model
            logger.info(f"Modelo cargado desde: {filepath}")
            
            # Intentar cargar parámetros
            params_file = os.path.splitext(filepath)[0] + "_params.json"
            if os.path.exists(params_file):
                params = pd.read_json(params_file, typ='series').to_dict()
                self.best_params = params
                logger.info(f"Parámetros cargados desde: {params_file}")
                
            return model
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            return None
