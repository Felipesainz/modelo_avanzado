"""
Procesamiento Avanzado de Datos para Forecasting
================================================
Módulo para limpieza de datos, detección de outliers, transformaciones
y generación de características avanzadas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.robust import mad
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import RobustScaler, PowerTransformer
import os
import logging
from config import *

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "data_processor.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataProcessor")

class DataProcessor:
    """Clase para procesamiento avanzado de datos de series temporales."""
    
    def __init__(self, file_path=DATA_FILE, target_column=TARGET_COLUMN):
        """
        Inicializa el procesador de datos.
        
        Args:
            file_path: Ruta al archivo CSV de datos
            target_column: Nombre de la columna objetivo
        """
        self.file_path = file_path
        self.target_column = target_column
        self.df_raw = None
        self.df_clean = None
        self.df_features = None
        self.df_transformed = None
        self.transformers = {}
        self.outliers = None
        self.best_transformation = None
        
    def load_data(self):
        """Carga los datos desde el archivo CSV."""
        logger.info(f"Cargando datos desde {self.file_path}")
        
        try:
            df = pd.read_csv(self.file_path)
            df['mes'] = pd.to_datetime(df['mes'])
            df.set_index('mes', inplace=True)
            
            self.df_raw = df
            logger.info(f"Datos cargados: {len(df)} registros, {df.shape[1]} columnas")
            logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
            
            # Estadísticas básicas
            logger.info("\nEstadísticas descriptivas:")
            logger.info(f"\n{df.describe()}")
            
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def detect_outliers(self, method='mad', threshold=OUTLIER_THRESHOLD, plot=True):
        """
        Detecta outliers en la serie temporal usando métodos robustos.
        
        Args:
            method: Método de detección ('mad', 'zscore', 'iqr')
            threshold: Umbral para detección
            plot: Si se debe generar un gráfico
            
        Returns:
            DataFrame con los outliers detectados
        """
        logger.info(f"Detectando outliers usando método '{method}' con umbral {threshold}")
        
        if self.df_raw is None:
            self.load_data()
            
        series = self.df_raw[self.target_column]
        outliers_idx = None
        
        if method == 'mad':
            # Método MAD (Mediana de Desviación Absoluta) - robusto a outliers
            median = np.median(series)
            mad_value = mad(series)
            z_scores = 0.6745 * (series - median) / mad_value
            outliers_idx = abs(z_scores) > threshold
            
        elif method == 'zscore':
            # Z-score tradicional
            z_scores = (series - series.mean()) / series.std()
            outliers_idx = abs(z_scores) > threshold
            
        elif method == 'iqr':
            # Rango intercuartil
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers_idx = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        
        outliers = series[outliers_idx]
        self.outliers = outliers
        
        # Registrar resultados
        if len(outliers) > 0:
            logger.info(f"Se detectaron {len(outliers)} outliers ({len(outliers)/len(series)*100:.1f}% de los datos)")
            logger.info(f"Outliers: \n{outliers}")
        else:
            logger.info("No se detectaron outliers significativos.")
        
        # Generar gráfico
        if plot and len(outliers) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(series.index, series, 'b-', label='Serie original')
            plt.scatter(outliers.index, outliers, color='red', s=50, label='Outliers')
            plt.title(f'Detección de Outliers (Método: {method}, Umbral: {threshold})', fontsize=13)
            plt.xlabel('Fecha')
            plt.ylabel(self.target_column)
            plt.legend()
            plt.tight_layout()
            
            outlier_plot_path = os.path.join(PLOTS_DIR, f"outliers_{method}.png")
            plt.savefig(outlier_plot_path)
            plt.close()
            logger.info(f"Gráfico de outliers guardado en: {outlier_plot_path}")
        
        return outliers
    
    def clean_outliers(self, method='interpolate'):
        """
        Limpia los outliers detectados usando diferentes métodos.
        
        Args:
            method: Método de limpieza ('interpolate', 'median', 'mean', 'rolling')
            
        Returns:
            DataFrame con los outliers limpiados
        """
        logger.info(f"Limpiando outliers usando método '{method}'")
        
        if self.outliers is None:
            self.detect_outliers()
            
        if len(self.outliers) == 0:
            logger.info("No hay outliers para limpiar.")
            self.df_clean = self.df_raw.copy()
            return self.df_clean
            
        # Crear copia para no modificar los datos originales
        df_clean = self.df_raw.copy()
        
        # Guardar los valores originales para comparación
        original_values = df_clean.loc[self.outliers.index, self.target_column].copy()
        
        # Aplicar método de limpieza
        if method == 'interpolate':
            # Reemplazar outliers con NaN y luego interpolar
            df_clean.loc[self.outliers.index, self.target_column] = np.nan
            df_clean[self.target_column] = df_clean[self.target_column].interpolate(method='time')
            
        elif method == 'median':
            # Reemplazar con la mediana
            median_value = df_clean[self.target_column].median()
            df_clean.loc[self.outliers.index, self.target_column] = median_value
            
        elif method == 'mean':
            # Reemplazar con la media
            mean_value = df_clean[self.target_column].mean()
            df_clean.loc[self.outliers.index, self.target_column] = mean_value
            
        elif method == 'rolling':
            # Reemplazar con media móvil centrada (excluyendo el outlier)
            window_size = 5
            for idx in self.outliers.index:
                # Obtener ventana alrededor del outlier
                window_start = max(0, df_clean.index.get_loc(idx) - window_size // 2)
                window_end = min(len(df_clean), df_clean.index.get_loc(idx) + window_size // 2 + 1)
                
                # Extraer valores en la ventana, excluyendo otros outliers
                window_values = df_clean.iloc[window_start:window_end][self.target_column].copy()
                window_values = window_values[~window_values.index.isin(self.outliers.index)]
                
                if len(window_values) > 0:
                    # Reemplazar con la media de la ventana
                    df_clean.loc[idx, self.target_column] = window_values.mean()
                else:
                    # Si no hay valores en la ventana, usar media global
                    df_clean.loc[idx, self.target_column] = df_clean[self.target_column].mean()
                    
        # Comparar valores originales vs. limpiados
        cleaned_values = df_clean.loc[self.outliers.index, self.target_column]
        comparison = pd.DataFrame({
            'Original': original_values,
            'Limpiado': cleaned_values,
            'Diferencia': original_values - cleaned_values,
            'Diferencia %': ((original_values - cleaned_values) / original_values) * 100
        })
        
        logger.info(f"Comparación de valores originales vs. limpiados:\n{comparison}")
        
        # Generar gráfico antes/después
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.df_raw.index, self.df_raw[self.target_column], 'b-', label='Original')
        plt.scatter(self.outliers.index, self.outliers, color='red', s=50, label='Outliers')
        plt.title('Serie Original con Outliers', fontsize=13)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(df_clean.index, df_clean[self.target_column], 'g-', label='Limpiada')
        plt.title(f'Serie con Outliers Limpiados (Método: {method})', fontsize=13)
        plt.legend()
        
        plt.tight_layout()
        clean_plot_path = os.path.join(PLOTS_DIR, f"outliers_cleaned_{method}.png")
        plt.savefig(clean_plot_path)
        plt.close()
        
        logger.info(f"Gráfico de comparación guardado en: {clean_plot_path}")
        
        self.df_clean = df_clean
        return df_clean
    
    def check_stationarity(self):
        """
        Verifica la estacionariedad de la serie temporal.
        
        Returns:
            dict: Resultados de las pruebas
        """
        logger.info("Verificando estacionariedad de la serie temporal")
        
        df = self.df_clean if self.df_clean is not None else self.df_raw
        series = df[self.target_column].dropna()
        
        results = {}
        
        # Prueba ADF - H0: La serie no es estacionaria
        try:
            adf_result = adfuller(series)
            adf_stat, adf_pvalue = adf_result[0], adf_result[1]
            
            results['adf_stat'] = adf_stat
            results['adf_pvalue'] = adf_pvalue
            results['adf_stationary'] = adf_pvalue < 0.05
            
            logger.info(f"Prueba ADF - Estadístico: {adf_stat:.4f}, p-valor: {adf_pvalue:.4f}")
            logger.info(f"ADF Conclusión: {'Estacionaria' if adf_pvalue < 0.05 else 'No estacionaria'}")
        except Exception as e:
            logger.warning(f"Error en prueba ADF: {e}")
        
        # Prueba KPSS - H0: La serie es estacionaria
        try:
            kpss_result = kpss(series)
            kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
            
            results['kpss_stat'] = kpss_stat
            results['kpss_pvalue'] = kpss_pvalue
            results['kpss_stationary'] = kpss_pvalue >= 0.05
            
            logger.info(f"Prueba KPSS - Estadístico: {kpss_stat:.4f}, p-valor: {kpss_pvalue:.4f}")
            logger.info(f"KPSS Conclusión: {'Estacionaria' if kpss_pvalue >= 0.05 else 'No estacionaria'}")
        except Exception as e:
            logger.warning(f"Error en prueba KPSS: {e}")
        
        # Consenso basado en ambas pruebas
        if 'adf_stationary' in results and 'kpss_stationary' in results:
            if results['adf_stationary'] and results['kpss_stationary']:
                consensus = "Fuertemente estacionaria"
            elif results['adf_stationary'] and not results['kpss_stationary']:
                consensus = "Tendencia estacionaria"
            elif not results['adf_stationary'] and results['kpss_stationary']:
                consensus = "Diferencia estacionaria"
            else:
                consensus = "No estacionaria"
                
            results['consensus'] = consensus
            logger.info(f"Consenso de estacionariedad: {consensus}")
        
        return results
    
    def decompose_series(self, period=12):
        """
        Descompone la serie en tendencia, estacionalidad y residuo.
        
        Args:
            period: Período para la descomposición (default: 12 para datos mensuales)
            
        Returns:
            Componentes de la descomposición
        """
        logger.info(f"Descomponiendo serie con período {period}")
        
        df = self.df_clean if self.df_clean is not None else self.df_raw
        series = df[self.target_column]
        
        try:
            # Verificar que hay suficientes datos
            if len(series) < 2 * period:
                logger.warning(f"Datos insuficientes para descomposición con período {period}")
                return None
            
            # Descomponer serie
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            # Guardar gráfico
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0], title='Serie Original')
            decomposition.trend.plot(ax=axes[1], title='Tendencia')
            decomposition.seasonal.plot(ax=axes[2], title='Estacionalidad')
            decomposition.resid.plot(ax=axes[3], title='Residuos')
            plt.tight_layout()
            
            decomp_path = os.path.join(PLOTS_DIR, f"decomposition_p{period}.png")
            plt.savefig(decomp_path)
            plt.close()
            
            logger.info(f"Descomposición guardada en: {decomp_path}")
            
            # Análisis de estacionalidad mensual
            df_month = pd.DataFrame()
            df_month['month'] = series.index.month
            df_month['value'] = series.values
            monthly_avg = df_month.groupby('month')['value'].mean()
            
            plt.figure(figsize=(10, 6))
            monthly_avg.plot(kind='bar')
            plt.title('Patrón Estacional Mensual', fontsize=13)
            plt.xlabel('Mes')
            plt.ylabel('Valor Promedio')
            plt.xticks(range(12), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
            plt.tight_layout()
            
            seasonal_path = os.path.join(PLOTS_DIR, "seasonal_pattern.png")
            plt.savefig(seasonal_path)
            plt.close()
            
            logger.info(f"Patrón estacional guardado en: {seasonal_path}")
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Error en descomposición: {e}")
            return None
    
    def evaluate_transformations(self):
        """
        Evalúa diferentes transformaciones para la variable objetivo.
        
        Returns:
            Mejor transformación basada en normalidad
        """
        logger.info("Evaluando transformaciones para la variable objetivo")
        
        df = self.df_clean if self.df_clean is not None else self.df_raw
        y = df[self.target_column].values.reshape(-1, 1)
        
        transformations = {}
        normality_scores = {}
        
        # Evaluar diferentes transformaciones
        # 1. Original (sin transformación)
        transformations['none'] = y
        
        # 2. Log(1+x)
        transformations['log'] = np.log1p(y)
        
        # 3. Raíz cuadrada
        transformations['sqrt'] = np.sqrt(y)
        
        # 4. Box-Cox
        # Requiere valores estrictamente positivos
        if np.all(y > 0):
            try:
                y_boxcox, lambda_boxcox = stats.boxcox(y.flatten())
                transformations['boxcox'] = y_boxcox.reshape(-1, 1)
                self.transformers['boxcox_lambda'] = lambda_boxcox
                logger.info(f"Lambda óptima para Box-Cox: {lambda_boxcox:.4f}")
            except Exception as e:
                logger.warning(f"Error en transformación Box-Cox: {e}")
        else:
            logger.warning("Box-Cox no aplicable: datos no positivos")
        
        # 5. Yeo-Johnson (funciona con negativos)
        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            y_yj = pt.fit_transform(y)
            transformations['yeojohnson'] = y_yj
            self.transformers['yeojohnson'] = pt
        except Exception as e:
            logger.warning(f"Error en transformación Yeo-Johnson: {e}")
        
        # Evaluar normalidad usando test de Shapiro-Wilk
        # y gráficos Q-Q
        for name, data in transformations.items():
            try:
                # Test de Shapiro-Wilk
                shapiro_stat, shapiro_p = stats.shapiro(data.flatten())
                normality_scores[name] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    # Mayor p-valor indica mejor normalidad
                    'score': shapiro_p
                }
                logger.info(f"Shapiro-Wilk para '{name}': estadístico={shapiro_stat:.4f}, p-valor={shapiro_p:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluando normalidad para '{name}': {e}")
        
        # Graficar histogramas y Q-Q plots
        fig, axes = plt.subplots(len(transformations), 2, figsize=(15, 4*len(transformations)))
        
        for i, (name, data) in enumerate(transformations.items()):
            # Título para la fila
            score_text = f"(Shapiro p={normality_scores[name]['shapiro_p']:.4f})" if name in normality_scores else ""
            row_title = f"Transformación: {name} {score_text}"
            axes[i, 0].set_title(f"{row_title} - Histograma")
            axes[i, 1].set_title(f"{row_title} - Q-Q Plot")
            
            # Histograma
            sns.histplot(data.flatten(), kde=True, ax=axes[i, 0])
            
            # Q-Q Plot
            stats.probplot(data.flatten(), dist="norm", plot=axes[i, 1])
        
        plt.tight_layout()
        transform_path = os.path.join(PLOTS_DIR, "transformations.png")
        plt.savefig(transform_path)
        plt.close()
        
        logger.info(f"Gráficos de transformaciones guardados en: {transform_path}")
        
        # Seleccionar mejor transformación basada en la prueba de normalidad
        best_transform = max(normality_scores.items(), key=lambda x: x[1]['score'])
        logger.info(f"Mejor transformación: '{best_transform[0]}' con p-valor {best_transform[1]['score']:.4f}")
        
        self.best_transformation = best_transform[0]
        return best_transform[0]
    
    def apply_transformation(self, method=None):
        """
        Aplica la transformación seleccionada a la variable objetivo.
        
        Args:
            method: Método de transformación (si None, se usa la mejor encontrada)
            
        Returns:
            DataFrame con la variable objetivo transformada
        """
        if method is None:
            if self.best_transformation is None:
                self.evaluate_transformations()
            method = self.best_transformation
        
        logger.info(f"Aplicando transformación '{method}' a la variable objetivo")
        
        df = self.df_clean if self.df_clean is not None else self.df_raw
        df_transformed = df.copy()
        
        # Aplicar transformación
        if method == 'none':
            # No hacer nada
            pass
        elif method == 'log':
            df_transformed[self.target_column] = np.log1p(df[self.target_column])
        elif method == 'sqrt':
            df_transformed[self.target_column] = np.sqrt(df[self.target_column])
        elif method == 'boxcox' and 'boxcox_lambda' in self.transformers:
            lambda_val = self.transformers['boxcox_lambda']
            df_transformed[self.target_column] = stats.boxcox(df[self.target_column], lambda_val)
        elif method == 'yeojohnson' and 'yeojohnson' in self.transformers:
            transformer = self.transformers['yeojohnson']
            df_transformed[self.target_column] = transformer.transform(df[self.target_column].values.reshape(-1, 1))
        else:
            logger.warning(f"Transformación '{method}' no disponible o no configurada")
            return df
        
        self.df_transformed = df_transformed
        return df_transformed
    
    def inverse_transform(self, series, method=None):
        """
        Aplica la transformación inversa a los valores.
        
        Args:
            series: Serie o array a des-transformar
            method: Método de transformación usado
            
        Returns:
            Valores originales (des-transformados)
        """
        if method is None:
            method = self.best_transformation
            
        if method == 'none':
            return series
        elif method == 'log':
            return np.expm1(series)
        elif method == 'sqrt':
            return np.square(series)
        elif method == 'boxcox' and 'boxcox_lambda' in self.transformers:
            lambda_val = self.transformers['boxcox_lambda']
            return stats.inv_boxcox(series, lambda_val)
        elif method == 'yeojohnson' and 'yeojohnson' in self.transformers:
            transformer = self.transformers['yeojohnson']
            if hasattr(transformer, 'inverse_transform'):
                return transformer.inverse_transform(series.reshape(-1, 1)).flatten()
            else:
                logger.warning("Transformación inversa no disponible para Yeo-Johnson")
                return series
        else:
            logger.warning(f"Transformación inversa para '{method}' no implementada")
            return series
