"""
Script Principal de Forecasting Avanzado
==========================================
Integra todos los módulos para ejecutar el flujo completo
de procesamiento de datos, ingeniería de características, 
optimización de modelos y ensamblado.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime
import argparse

# Agregar directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos propios
from config import *
from data_processor import DataProcessor
from feature_engineering import FeatureEngineer
from model_optimizer import ModelOptimizer
from ensemble_models import TimeSeriesEnsemble

# Configurar logging principal
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

def create_directories():
    """Crea las carpetas necesarias para la ejecución."""
    directories = [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Creada carpeta: {directory}")

def load_data(file_path):
    """
    Carga los datos desde el archivo CSV.
    
    Args:
        file_path: Ruta al archivo de datos
        
    Returns:
        DataFrame con los datos cargados
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Datos cargados correctamente: {file_path}")
        logger.info(f"Dimensiones: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        sys.exit(1)

def execute_full_pipeline(data_path, target_col='demanda', date_col='fecha', 
                         test_size=0.2, use_ensemble=True, save_models=True,
                         n_trials=100, cv_folds=5):
    """
    Ejecuta el pipeline completo de forecasting.
    
    Args:
        data_path: Ruta al archivo de datos
        target_col: Nombre de la columna objetivo
        date_col: Nombre de la columna de fecha
        test_size: Fracción de datos para testing
        use_ensemble: Si se debe usar ensemble o solo XGBoost
        save_models: Si se deben guardar los modelos entrenados
        n_trials: Número de pruebas para optimización
        cv_folds: Número de folds para validación cruzada
    
    Returns:
        Métricas de evaluación
    """
    # Iniciar temporizador
    start_time = time.time()
    logger.info("Iniciando pipeline completo de forecasting")
    
    # Crear directorios
    create_directories()
    
    # 1. Cargar datos
    data = load_data(data_path)
    logger.info(f"Ejemplo de datos:\n{data.head()}")
    
    # 2. Procesar datos
    logger.info("Iniciando procesamiento de datos...")
    data_processor = DataProcessor(data, date_column=date_col, target_column=target_col)
    data_processor.run_full_processing()
    processed_data = data_processor.get_processed_data()
    logger.info("Procesamiento de datos completado")
    
    # 3. Ingeniería de características
    logger.info("Iniciando ingeniería de características...")
    feature_engineer = FeatureEngineer(processed_data, date_column=date_col, target_column=target_col)
    feature_engineer.create_all_features()
    feature_df = feature_engineer.get_feature_df()
    logger.info(f"Características generadas: {feature_df.shape[1]}")
    
    # 4. Dividir en train y test
    X_train, X_test, y_train, y_test = feature_engineer.train_test_split(test_size=test_size)
    logger.info(f"Datos de entrenamiento: {X_train.shape}, Datos de prueba: {X_test.shape}")
    
    # 5. Optimizar modelo XGBoost
    logger.info("Iniciando optimización de modelo XGBoost...")
    optimizer = ModelOptimizer(
        X_train, y_train, X_test, y_test, 
        cv_folds=cv_folds, 
        n_trials=n_trials
    )
    
    best_params = optimizer.optimize_xgboost()
    logger.info(f"Mejores parámetros: {best_params}")
    
    xgb_model = optimizer.train_best_model()
    xgb_metrics = optimizer.evaluate_model()
    logger.info(f"Métricas XGBoost: {xgb_metrics}")
    
    if save_models:
        xgb_path = optimizer.save_model(filename="xgboost_best_model.pkl")
        logger.info(f"Modelo XGBoost guardado en: {xgb_path}")
    
    # 6. Ensemble (opcional)
    if use_ensemble:
        logger.info("Iniciando construcción de ensemble...")
        ensemble = TimeSeriesEnsemble(X_train, y_train, X_test, y_test)
        ensemble.create_base_models(n_models=5)
        
        # Probar diferentes tipos de ensemble
        logger.info("Construyendo ensemble de votación...")
        voting_ensemble = ensemble.build_voting_ensemble()
        voting_metrics = ensemble.metrics
        
        logger.info("Construyendo ensemble stacking...")
        stacking_ensemble = ensemble.build_stacking_ensemble()
        stacking_metrics = ensemble.metrics
        
        logger.info("Construyendo ensemble ponderado...")
        weighted_ensemble = ensemble.build_weighted_ensemble()
        weighted_metrics = ensemble.metrics
        
        # Seleccionar el mejor ensemble
        ensemble_metrics = [
            ("voting", voting_metrics),
            ("stacking", stacking_metrics),
            ("weighted", weighted_metrics)
        ]
        
        best_ensemble_name, best_ensemble_metrics = min(
            ensemble_metrics, 
            key=lambda x: x[1]['mae'] if x[1] is not None else float('inf')
        )
        
        logger.info(f"Mejor ensemble: {best_ensemble_name}")
        logger.info(f"Métricas del mejor ensemble: {best_ensemble_metrics}")
        
        if save_models and best_ensemble_metrics is not None:
            ensemble_path = ensemble.save_ensemble(filename=f"ensemble_{best_ensemble_name}.pkl")
            logger.info(f"Mejor ensemble guardado en: {ensemble_path}")
        
        # Comparar con XGBoost base
        if xgb_metrics is not None and best_ensemble_metrics is not None:
            improvement = (xgb_metrics['mae'] - best_ensemble_metrics['mae']) / xgb_metrics['mae'] * 100
            logger.info(f"Mejora porcentual del ensemble sobre XGBoost: {improvement:.2f}%")
            
            # Crear tabla comparativa
            metrics_comparison = pd.DataFrame({
                'XGBoost': xgb_metrics,
                f'Ensemble ({best_ensemble_name})': best_ensemble_metrics
            })
            metrics_comparison.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
            
            # Visualizar comparación
            plt.figure(figsize=(10, 6))
            metrics_comparison.iloc[:3].plot(kind='bar')
            plt.title('Comparación de Métricas: XGBoost vs Ensemble')
            plt.ylabel('Valor (menor es mejor)')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(PLOTS_DIR, "metrics_comparison.png"))
            plt.close()
    
    # 7. Tiempo de ejecución
    duration = time.time() - start_time
    logger.info(f"Tiempo total de ejecución: {duration:.2f} segundos ({duration/60:.2f} minutos)")
    
    # 8. Resumen final
    logger.info("\n========== RESUMEN FINAL ==========")
    logger.info(f"Datos procesados: {processed_data.shape}")
    logger.info(f"Características generadas: {feature_df.shape[1]}")
    logger.info(f"Métrica MAE XGBoost: {xgb_metrics['mae'] if xgb_metrics else 'N/A'}")
    
    if use_ensemble and best_ensemble_metrics:
        logger.info(f"Métrica MAE Ensemble: {best_ensemble_metrics['mae']}")
        logger.info(f"Mejor tipo de ensemble: {best_ensemble_name}")
    
    # Generar reporte
    generate_report(
        data_processor=data_processor,
        feature_engineer=feature_engineer,
        optimizer=optimizer,
        ensemble=ensemble if use_ensemble else None,
        xgb_metrics=xgb_metrics,
        best_ensemble_metrics=best_ensemble_metrics if use_ensemble and best_ensemble_metrics else None,
        best_ensemble_name=best_ensemble_name if use_ensemble and best_ensemble_metrics else None,
        execution_time=duration
    )
    
    return xgb_metrics, best_ensemble_metrics if use_ensemble else None

def generate_report(data_processor, feature_engineer, optimizer, ensemble=None,
                   xgb_metrics=None, best_ensemble_metrics=None, best_ensemble_name=None,
                   execution_time=None):
    """
    Genera un reporte HTML con los resultados del pipeline.
    
    Args:
        data_processor: Instancia de DataProcessor
        feature_engineer: Instancia de FeatureEngineer
        optimizer: Instancia de ModelOptimizer
        ensemble: Instancia de TimeSeriesEnsemble (opcional)
        xgb_metrics: Métricas de XGBoost
        best_ensemble_metrics: Métricas del mejor ensemble
        best_ensemble_name: Nombre del mejor ensemble
        execution_time: Tiempo de ejecución total
    """
    try:
        import jinja2
        
        # Crear directorio para reporte
        if not os.path.exists(os.path.join(RESULTS_DIR, "report")):
            os.makedirs(os.path.join(RESULTS_DIR, "report"))
        
        # Copiar imágenes para el reporte
        import shutil
        for file in os.listdir(PLOTS_DIR):
            if file.endswith('.png'):
                shutil.copy(
                    os.path.join(PLOTS_DIR, file),
                    os.path.join(RESULTS_DIR, "report", file)
                )
        
        # Template HTML
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Forecasting - Cluster 1</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { color: #2980b9; margin-top: 30px; }
                h3 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { text-align: left; padding: 12px; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .metric-card { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .image-container { margin: 20px 0; text-align: center; }
                .image-container img { max-width: 100%; height: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                .highlight { color: #e74c3c; font-weight: bold; }
                .container { display: flex; flex-wrap: wrap; }
                .half { flex: 1; min-width: 45%; margin: 10px; }
                footer { margin-top: 50px; text-align: center; font-size: 0.8em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <h1>Reporte de Forecasting Avanzado - Cluster 1</h1>
            <p>Fecha de generación: {{ date }}</p>
            
            <h2>Resumen Ejecutivo</h2>
            <div class="container">
                <div class="half metric-card">
                    <h3>Métricas XGBoost</h3>
                    {% if xgb_metrics %}
                    <ul>
                        <li>MAE: <span class="highlight">{{ xgb_metrics.mae|round(4) }}</span></li>
                        <li>RMSE: {{ xgb_metrics.rmse|round(4) }}</li>
                        <li>MAPE: {{ xgb_metrics.mape|round(2) }}%</li>
                        <li>R²: {{ xgb_metrics.r2|round(4) }}</li>
                    </ul>
                    {% else %}
                    <p>No disponible</p>
                    {% endif %}
                </div>
                
                <div class="half metric-card">
                    <h3>Métricas Ensemble{% if best_ensemble_name %} ({{ best_ensemble_name }}){% endif %}</h3>
                    {% if best_ensemble_metrics %}
                    <ul>
                        <li>MAE: <span class="highlight">{{ best_ensemble_metrics.mae|round(4) }}</span></li>
                        <li>RMSE: {{ best_ensemble_metrics.rmse|round(4) }}</li>
                        <li>MAPE: {{ best_ensemble_metrics.mape|round(2) }}%</li>
                        <li>R²: {{ best_ensemble_metrics.r2|round(4) }}</li>
                    </ul>
                    {% else %}
                    <p>No disponible</p>
                    {% endif %}
                </div>
            </div>
            
            {% if xgb_metrics and best_ensemble_metrics %}
            <p>Mejora del ensemble sobre XGBoost: <span class="highlight">{{ ((xgb_metrics.mae - best_ensemble_metrics.mae) / xgb_metrics.mae * 100)|round(2) }}%</span></p>
            {% endif %}
            
            <div class="image-container">
                <img src="metrics_comparison.png" alt="Comparación de métricas" width="700">
            </div>
            
            <h2>Procesamiento de Datos</h2>
            <p>Se procesaron {{ data_processor.data.shape[0] }} registros con {{ data_processor.data.shape[1] }} columnas.</p>
            
            {% if outliers_count %}
            <p>Se detectaron {{ outliers_count }} outliers ({{ (outliers_count/data_processor.data.shape[0]*100)|round(2) }}% de los datos).</p>
            {% endif %}
            
            <div class="image-container">
                <img src="data_processing.png" alt="Procesamiento de datos" width="700">
            </div>
            
            <h2>Ingeniería de Características</h2>
            <p>Se generaron {{ feature_engineer.feature_df.shape[1] }} características a partir de los datos procesados.</p>
            
            <h3>Top 10 Características más Importantes</h3>
            {% if feature_importances %}
            <table>
                <tr>
                    <th>Característica</th>
                    <th>Importancia</th>
                </tr>
                {% for feature, importance in feature_importances %}
                <tr>
                    <td>{{ feature }}</td>
                    <td>{{ importance|round(4) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>Información no disponible</p>
            {% endif %}
            
            <div class="image-container">
                <img src="feature_importances.png" alt="Importancia de características" width="700">
            </div>
            
            <h2>Optimización de Modelo</h2>
            <p>Se realizaron {{ optimizer.n_trials }} pruebas para encontrar los mejores hiperparámetros.</p>
            
            <h3>Mejores Hiperparámetros</h3>
            {% if best_params %}
            <table>
                <tr>
                    <th>Parámetro</th>
                    <th>Valor</th>
                </tr>
                {% for param, value in best_params.items() %}
                <tr>
                    <td>{{ param }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>Información no disponible</p>
            {% endif %}
            
            <div class="image-container">
                <img src="optimization_history.png" alt="Historia de optimización" width="700">
            </div>
            
            <div class="image-container">
                <img src="param_importances.png" alt="Importancia de hiperparámetros" width="700">
            </div>
            
            <h2>Predicciones</h2>
            
            <div class="container">
                <div class="half">
                    <h3>XGBoost</h3>
                    <div class="image-container">
                        <img src="predictions.png" alt="Predicciones XGBoost" width="400">
                    </div>
                </div>
                
                {% if ensemble %}
                <div class="half">
                    <h3>Ensemble</h3>
                    <div class="image-container">
                        <img src="ensemble_predictions.png" alt="Predicciones Ensemble" width="400">
                    </div>
                </div>
                {% endif %}
            </div>
            
            <h2>Tiempo de Ejecución</h2>
            <p>Tiempo total: {{ execution_time|round(2) }} segundos ({{ (execution_time/60)|round(2) }} minutos)</p>
            
            <footer>
                <p>Reporte generado automáticamente por el pipeline de forecasting avanzado</p>
            </footer>
        </body>
        </html>
        """
        
        # Preparar datos para el template
        template_data = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'xgb_metrics': xgb_metrics,
            'best_ensemble_metrics': best_ensemble_metrics,
            'best_ensemble_name': best_ensemble_name,
            'data_processor': data_processor,
            'feature_engineer': feature_engineer,
            'optimizer': optimizer,
            'outliers_count': data_processor.outliers_count if hasattr(data_processor, 'outliers_count') else None,
            'best_params': optimizer.best_params,
            'feature_importances': optimizer.feature_importances.head(10).values if hasattr(optimizer, 'feature_importances') else None,
            'ensemble': ensemble,
            'execution_time': execution_time
        }
        
        # Renderizar template
        template = jinja2.Template(template_str)
        html_output = template.render(**template_data)
        
        # Guardar reporte
        report_path = os.path.join(RESULTS_DIR, "report", "index.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
            
        logger.info(f"Reporte generado en: {report_path}")
    except Exception as e:
        logger.error(f"Error al generar reporte: {e}")

def parse_arguments():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Ejecutar pipeline de forecasting avanzado')
    
    parser.add_argument('--data', type=str, required=True,
                      help='Ruta al archivo CSV con los datos')
    
    parser.add_argument('--target', type=str, default='demanda',
                      help='Nombre de la columna objetivo (default: demanda)')
    
    parser.add_argument('--date', type=str, default='fecha',
                      help='Nombre de la columna de fecha (default: fecha)')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Fracción de datos para testing (default: 0.2)')
    
    parser.add_argument('--no_ensemble', action='store_true',
                      help='Desactivar uso de ensemble')
    
    parser.add_argument('--no_save', action='store_true',
                      help='No guardar los modelos entrenados')
    
    parser.add_argument('--trials', type=int, default=50,
                      help='Número de pruebas para optimización (default: 50)')
    
    parser.add_argument('--cv_folds', type=int, default=5,
                      help='Número de folds para validación cruzada (default: 5)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Analizar argumentos
    args = parse_arguments()
    
    # Ejecutar pipeline
    execute_full_pipeline(
        data_path=args.data,
        target_col=args.target,
        date_col=args.date,
        test_size=args.test_size,
        use_ensemble=not args.no_ensemble,
        save_models=not args.no_save,
        n_trials=args.trials,
        cv_folds=args.cv_folds
    )
