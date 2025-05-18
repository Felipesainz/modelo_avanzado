"""
Modelos Ensemble Avanzados para Forecasting
===========================================
Implementa técnicas de ensemble para mejorar la precisión y
robustez de las predicciones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import os
import pickle
import logging
from config import *

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "ensemble_models.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnsembleModels")

class TimeSeriesEnsemble:
    """Clase para modelos ensemble avanzados para series temporales."""
    
    def __init__(self, X_train, y_train, X_test=None, y_test=None, random_state=RANDOM_SEED):
        """
        Inicializa el ensemble.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_test: Características de prueba (opcional)
            y_test: Target de prueba (opcional)
            random_state: Semilla aleatoria
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.models = []
        self.ensemble_model = None
        self.metrics = None
        
    def create_base_models(self, n_models=5):
        """
        Crea modelos base para el ensemble con diferentes configuraciones.
        
        Args:
            n_models: Número de modelos a crear
            
        Returns:
            Lista de modelos base
        """
        logger.info(f"Creando {n_models} modelos base para ensemble")
        
        models = []
        
        # 1. XGBoost con configuraciones diversas
        # Modelo 1: Enfocado en prevenir sobreajuste
        models.append(
            ('xgb1', XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=self.random_state
            ))
        )
        
        # Modelo 2: Más profundo, bueno para capturar patrones complejos
        if n_models >= 2:
            models.append(
                ('xgb2', XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.03,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=0.8,
                    random_state=self.random_state + 1
                ))
            )
        
        # Modelo 3: Más árboles pero poco profundos (reduce varianza)
        if n_models >= 3:
            models.append(
                ('xgb3', XGBRegressor(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.01,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    random_state=self.random_state + 2
                ))
            )
            
        # 2. LightGBM con configuraciones diversas
        if n_models >= 4:
            models.append(
                ('lgbm1', LGBMRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=self.random_state + 3
                ))
            )
            
        if n_models >= 5:
            models.append(
                ('lgbm2', LGBMRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=0.5,
                    random_state=self.random_state + 4
                ))
            )
        
        # Guardar modelos
        self.models = models
        
        return models
    
    def build_voting_ensemble(self, weights=None):
        """
        Construye un ensemble de votación con los modelos base.
        
        Args:
            weights: Pesos para cada modelo base (si None, usa pesos iguales)
            
        Returns:
            Modelo ensemble de votación
        """
        logger.info("Construyendo ensemble de votación")
        
        if not self.models:
            self.create_base_models()
            
        # Si no se proporcionan pesos, usar pesos iguales
        if weights is None:
            weights = [1] * len(self.models)
            
        # Crear ensemble de votación
        voting_ensemble = VotingRegressor(
            estimators=self.models,
            weights=weights
        )
        
        # Entrenar ensemble
        voting_ensemble.fit(self.X_train, self.y_train)
        
        # Guardar ensemble
        self.ensemble_model = voting_ensemble
        
        # Evaluar si hay datos de test
        if self.X_test is not None and self.y_test is not None:
            self.evaluate_ensemble()
            
        return voting_ensemble
    
    def build_stacking_ensemble(self, meta_model=None):
        """
        Construye un ensemble de stacking con los modelos base.
        
        Args:
            meta_model: Modelo meta para combinar predicciones (si None, usa Ridge)
            
        Returns:
            Modelo ensemble de stacking
        """
        logger.info("Construyendo ensemble de stacking")
        
        if not self.models:
            self.create_base_models()
            
        # Si no se proporciona meta_model, usar Ridge
        if meta_model is None:
            meta_model = Ridge(alpha=1.0, random_state=self.random_state)
            
        # Crear ensemble de stacking
        stacking_ensemble = StackingRegressor(
            estimators=self.models,
            final_estimator=meta_model,
            cv=TimeSeriesSplit(n_splits=5)
        )
        
        # Entrenar ensemble
        stacking_ensemble.fit(self.X_train, self.y_train)
        
        # Guardar ensemble
        self.ensemble_model = stacking_ensemble
        
        # Evaluar si hay datos de test
        if self.X_test is not None and self.y_test is not None:
            self.evaluate_ensemble()
            
        return stacking_ensemble
    
    def build_weighted_ensemble(self):
        """
        Construye un ensemble ponderado optimizando los pesos mediante validación cruzada.
        
        Returns:
            Modelo ensemble ponderado
        """
        logger.info("Construyendo ensemble ponderado con optimización de pesos")
        
        if not self.models:
            self.create_base_models()
            
        # Crear modelos base entrenados
        base_models = []
        for name, model in self.models:
            model_copy = pickle.loads(pickle.dumps(model))  # Clonar modelo
            model_copy.fit(self.X_train, self.y_train)
            base_models.append((name, model_copy))
            
        # Función para validación cruzada temporal
        def time_series_cv_score(models, weights, X, y, n_splits=5):
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Predecir con cada modelo
                preds = []
                for _, model in models:
                    model_copy = pickle.loads(pickle.dumps(model))
                    model_copy.fit(X_train_cv, y_train_cv)
                    preds.append(model_copy.predict(X_val_cv))
                    
                # Combinación ponderada
                weighted_pred = np.zeros(len(y_val_cv))
                for i, pred in enumerate(preds):
                    weighted_pred += weights[i] * pred
                    
                weighted_pred /= np.sum(weights)  # Normalizar
                
                # Calcular métrica
                mae = mean_absolute_error(y_val_cv, weighted_pred)
                cv_scores.append(mae)
                
            return np.mean(cv_scores)
        
        # Optimización de pesos con Optuna
        def objective(trial):
            # Generar pesos entre 0 y 1
            weights = [trial.suggest_float(f'w{i}', 0.0, 1.0) for i in range(len(base_models))]
            
            # Normalizar pesos
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights = [w / weights_sum for w in weights]
            else:
                weights = [1.0 / len(weights) for _ in weights]  # Pesos iguales si sum=0
                
            # Calcular score
            cv_score = time_series_cv_score(base_models, weights, self.X_train, self.y_train)
            
            return cv_score
        
        # Crear estudio Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        # Obtener mejores pesos
        best_weights = [study.best_params[f'w{i}'] for i in range(len(base_models))]
        
        # Normalizar pesos
        best_weights_sum = np.sum(best_weights)
        if best_weights_sum > 0:
            best_weights = [w / best_weights_sum for w in best_weights]
        
        logger.info(f"Mejores pesos: {best_weights}")
        
        # Crear ensemble con los mejores pesos
        ensemble = self.build_voting_ensemble(weights=best_weights)
        
        return ensemble
    
    def build_bagging_ensemble(self, n_bags=5, sample_fraction=0.8):
        """
        Construye un ensemble mediante bagging temporal.
        
        Args:
            n_bags: Número de bags a crear
            sample_fraction: Fracción de datos a usar en cada bag
            
        Returns:
            Modelo ensemble bagging
        """
        logger.info(f"Construyendo ensemble bagging con {n_bags} bags")
        
        if not self.models:
            # Usar solo XGBoost para bagging
            base_model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state
            )
        else:
            # Usar el primer modelo como base
            _, base_model = self.models[0]
        
        # Crear bags (modelos entrenados en diferentes subconjuntos)
        bag_models = []
        
        n_samples = int(len(self.X_train) * sample_fraction)
        
        for i in range(n_bags):
            # Crear copia del modelo base
            model_copy = pickle.loads(pickle.dumps(base_model))
            
            # Seleccionar subconjunto temporal (respetando cronología)
            if sample_fraction < 1.0:
                # Muestra aleatoria, pero manteniendo orden cronológico
                start = np.random.randint(0, len(self.X_train) - n_samples)
                X_bag = self.X_train.iloc[start:start+n_samples]
                y_bag = self.y_train.iloc[start:start+n_samples]
            else:
                # Usar todos los datos
                X_bag = self.X_train
                y_bag = self.y_train
            
            # Entrenar modelo
            model_copy.fit(X_bag, y_bag)
            
            # Agregar a la lista
            bag_models.append((f'bag{i}', model_copy))
            
        # Crear ensemble de votación con pesos iguales
        bagging_ensemble = VotingRegressor(
            estimators=bag_models,
            weights=[1] * n_bags
        )
        
        # No es necesario entrenar, ya que los modelos individuales están entrenados
        
        # Guardar ensemble
        self.ensemble_model = bagging_ensemble
        self.models = bag_models
        
        # Evaluar si hay datos de test
        if self.X_test is not None and self.y_test is not None:
            self.evaluate_ensemble()
            
        return bagging_ensemble
    
    def evaluate_ensemble(self):
        """
        Evalúa el ensemble en el conjunto de prueba.
        
        Returns:
            dict: Métricas de evaluación
        """
        if self.ensemble_model is None:
            logger.warning("No hay ensemble entrenado para evaluar")
            return None
            
        if self.X_test is None or self.y_test is None:
            logger.warning("No hay datos de prueba para evaluación")
            return None
            
        logger.info("Evaluando ensemble en datos de prueba")
        
        # Predecir
        y_pred = self.ensemble_model.predict(self.X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        
        # Métricas adicionales
        mpe = np.mean((self.y_test - y_pred) / self.y_test) * 100
        r2 = 1 - (np.sum((self.y_test - y_pred)**2) / np.sum((self.y_test - np.mean(self.y_test))**2))
        
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAPE: {mape:.4f}%")
        logger.info(f"MPE: {mpe:.4f}%")
        logger.info(f"R²: {r2:.4f}")
        
        # Guardar métricas
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mpe': mpe,
            'r2': r2
        }
        self.metrics = metrics
        
        # Guardar en archivo
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_metrics.csv"), index=False)
        
        # Graficar predicciones vs reales
        self.plot_ensemble_predictions(y_pred)
        
        return metrics
    
    def plot_ensemble_predictions(self, y_pred):
        """
        Grafica predicciones del ensemble vs valores reales.
        
        Args:
            y_pred: Predicciones
        """
        if self.y_test is None:
            return
            
        # Crear DataFrame con resultados
        df_plot = pd.DataFrame({
            'y_real': self.y_test,
            'y_pred': y_pred
        })
        
        # Calcular error
        df_plot['error'] = df_plot['y_real'] - df_plot['y_pred']
        df_plot['error_abs'] = np.abs(df_plot['error'])
        df_plot['error_pct'] = np.abs(df_plot['error'] / df_plot['y_real']) * 100
        
        # Gráfico principal
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(df_plot.index, df_plot['y_real'], 'b-', label='Real')
        plt.plot(df_plot.index, df_plot['y_pred'], 'r--', label='Ensemble')
        plt.title('Predicciones Ensemble vs Valores Reales')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.bar(df_plot.index, df_plot['error'], color='g', alpha=0.6)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Error de Predicción')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(df_plot.index, df_plot['error_pct'], 'm-', alpha=0.6)
        plt.axhline(y=df_plot['error_pct'].mean(), color='k', linestyle='--', 
                   label=f'Media: {df_plot["error_pct"].mean():.2f}%')
        plt.title('Error Porcentual')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "ensemble_predictions.png"))
        plt.close()
        
        # Guardar resultados
        df_plot.to_csv(os.path.join(RESULTS_DIR, "ensemble_predictions.csv"))
        
    def predict_with_ensemble(self, X):
        """
        Realiza predicciones con el ensemble.
        
        Args:
            X: Características para predicción
            
        Returns:
            Predicciones
        """
        if self.ensemble_model is None:
            logger.warning("No hay ensemble entrenado para predecir")
            return None
            
        return self.ensemble_model.predict(X)
    
    def predict_with_confidence(self, X, alpha=0.05):
        """
        Realiza predicciones con intervalos de confianza basados en la variabilidad
        de los modelos individuales.
        
        Args:
            X: Características para predicción
            alpha: Nivel de significancia (por defecto 0.05 para IC 95%)
            
        Returns:
            Predicciones con intervalos de confianza
        """
        if not self.models:
            logger.warning("No hay modelos base para calcular intervalos")
            return None
            
        # Predecir con cada modelo
        all_preds = []
        for _, model in self.models:
            pred = model.predict(X)
            all_preds.append(pred)
            
        # Convertir a array
        all_preds = np.array(all_preds)
        
        # Calcular estadísticas
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        # Calcular intervalos (usando distribución normal)
        from scipy import stats
        z = stats.norm.ppf(1 - alpha/2)
        lower_bound = mean_pred - z * std_pred
        upper_bound = mean_pred + z * std_pred
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_pred
        })
        
        return results
    
    def save_ensemble(self, filename=None):
        """
        Guarda el modelo ensemble.
        
        Args:
            filename: Nombre del archivo (si None, usa timestamp)
            
        Returns:
            Ruta al archivo guardado
        """
        if self.ensemble_model is None:
            logger.warning("No hay ensemble para guardar")
            return None
            
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_{timestamp}.pkl"
            
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Guardar ensemble
        with open(filepath, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
            
        logger.info(f"Ensemble guardado en: {filepath}")
        
        return filepath
    
    def load_ensemble(self, filepath):
        """
        Carga un modelo ensemble previamente guardado.
        
        Args:
            filepath: Ruta al archivo del ensemble
            
        Returns:
            Modelo ensemble cargado
        """
        try:
            with open(filepath, 'rb') as f:
                ensemble = pickle.load(f)
                
            self.ensemble_model = ensemble
            logger.info(f"Ensemble cargado desde: {filepath}")
            
            return ensemble
        except Exception as e:
            logger.error(f"Error al cargar ensemble: {e}")
            return None
