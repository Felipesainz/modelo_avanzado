"""
Ingeniería de Características Avanzada para Forecasting
======================================================
Implementa técnicas avanzadas de creación de características 
para mejorar la precisión del modelo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import os
import logging
from config import *

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "feature_engineering.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FeatureEngineering")

class FeatureEngineer:
    """Clase para ingeniería de características avanzada."""
    
    def __init__(self, df, target_column=TARGET_COLUMN):
        """
        Inicializa el ingeniero de características.
        
        Args:
            df: DataFrame con datos limpios
            target_column: Nombre de la columna objetivo
        """
        self.df = df
        self.target_column = target_column
        self.df_features = None
        self.feature_importances = None
        self.selected_features = None
        
    def create_time_features(self):
        """
        Crea características básicas basadas en el tiempo.
        
        Returns:
            DataFrame con características temporales
        """
        logger.info("Creando características temporales básicas")
        
        df_features = self.df.copy()
        
        # Características de calendario
        df_features['año'] = df_features.index.year
        df_features['mes'] = df_features.index.month
        df_features['trimestre'] = df_features.index.quarter
        df_features['día_año'] = df_features.index.dayofyear
        df_features['días_mes'] = df_features.index.days_in_month
        df_features['semana_año'] = df_features.index.isocalendar().week
        
        # Características derivadas
        df_features['es_fin_año'] = (df_features.index.month == 12).astype(int)
        df_features['es_inicio_año'] = (df_features.index.month == 1).astype(int)
        df_features['es_fin_trimestre'] = df_features.index.month.isin([3, 6, 9, 12]).astype(int)
        
        self.df_features = df_features
        return df_features
    
    def create_cyclical_features(self):
        """
        Crea características cíclicas usando codificación seno/coseno.
        
        Returns:
            DataFrame con características cíclicas añadidas
        """
        logger.info("Creando características cíclicas")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Codificación cíclica para mes (1-12)
        df_features['mes_sin'] = np.sin(2 * np.pi * df_features.index.month / 12)
        df_features['mes_cos'] = np.cos(2 * np.pi * df_features.index.month / 12)
        
        # Codificación cíclica para trimestre (1-4)
        df_features['trimestre_sin'] = np.sin(2 * np.pi * df_features['trimestre'] / 4)
        df_features['trimestre_cos'] = np.cos(2 * np.pi * df_features['trimestre'] / 4)
        
        # Codificación cíclica para día del año (1-366)
        df_features['día_año_sin'] = np.sin(2 * np.pi * df_features['día_año'] / 366)
        df_features['día_año_cos'] = np.cos(2 * np.pi * df_features['día_año'] / 366)
        
        # Armónicos adicionales para capturar patrones más complejos
        # Segundo armónico (6 meses)
        df_features['mes_sin_2'] = np.sin(4 * np.pi * df_features.index.month / 12) 
        df_features['mes_cos_2'] = np.cos(4 * np.pi * df_features.index.month / 12)
        
        # Tercer armónico (4 meses)
        df_features['mes_sin_3'] = np.sin(6 * np.pi * df_features.index.month / 12)
        df_features['mes_cos_3'] = np.cos(6 * np.pi * df_features.index.month / 12)
        
        # Cuarto armónico (3 meses)
        df_features['mes_sin_4'] = np.sin(8 * np.pi * df_features.index.month / 12)
        df_features['mes_cos_4'] = np.cos(8 * np.pi * df_features.index.month / 12)
        
        self.df_features = df_features
        return df_features
    
    def create_lag_features(self, max_lag=12):
        """
        Crea características de retraso (lag) para el target.
        
        Args:
            max_lag: Número máximo de lags a crear
            
        Returns:
            DataFrame con características de lag añadidas
        """
        logger.info(f"Creando características de lag (hasta {max_lag})")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Características lag (valores previos)
        for i in range(1, max_lag + 1):
            df_features[f'lag_{i}'] = df_features[self.target_column].shift(i)
            
        # Diferencias entre lags consecutivos
        for i in range(1, max_lag):
            df_features[f'lag_diff_{i}'] = df_features[f'lag_{i}'] - df_features[f'lag_{i+1}']
            
        # Aceleraciones (diferencias de diferencias)
        for i in range(1, max_lag-1):
            df_features[f'lag_accel_{i}'] = df_features[f'lag_diff_{i}'] - df_features[f'lag_diff_{i+1}']
            
        # Características de cambio porcentual
        for i in range(1, max_lag):
            df_features[f'lag_pct_{i}'] = (df_features[self.target_column].shift(i) / 
                                           df_features[self.target_column].shift(i+1) - 1) * 100
            
        self.df_features = df_features
        return df_features
    
    def create_rolling_features(self, windows=[3, 6, 12, 24], min_periods=1):
        """
        Crea características basadas en ventanas móviles.
        
        Args:
            windows: Lista de tamaños de ventana
            min_periods: Mínimo períodos requeridos
            
        Returns:
            DataFrame con características de ventana móvil añadidas
        """
        logger.info(f"Creando características de ventana móvil {windows}")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Para cada tamaño de ventana
        for window in windows:
            # Estadísticas básicas
            df_features[f'rolling_mean_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).mean()
            
            df_features[f'rolling_std_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).std()
            
            df_features[f'rolling_min_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).min()
            
            df_features[f'rolling_max_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).max()
            
            # Estadísticas más avanzadas
            df_features[f'rolling_kurt_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).kurt()
            
            df_features[f'rolling_skew_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).skew()
            
            df_features[f'rolling_median_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).median()
            
            df_features[f'rolling_q25_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).quantile(0.25)
            
            df_features[f'rolling_q75_{window}'] = df_features[self.target_column].rolling(
                window=window, min_periods=min_periods).quantile(0.75)
            
            # Coeficiente de variación
            df_features[f'rolling_cv_{window}'] = (df_features[f'rolling_std_{window}'] / 
                                                  df_features[f'rolling_mean_{window}'])
            
            # Rango intercuartil
            df_features[f'rolling_iqr_{window}'] = (df_features[f'rolling_q75_{window}'] - 
                                                   df_features[f'rolling_q25_{window}'])
            
        self.df_features = df_features
        return df_features
    
    def create_ewm_features(self, spans=[3, 6, 12, 24]):
        """
        Crea características basadas en medias móviles exponenciales.
        
        Args:
            spans: Lista de spans para EWM
            
        Returns:
            DataFrame con características EWM añadidas
        """
        logger.info(f"Creando características de media móvil exponencial {spans}")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Para cada span
        for span in spans:
            # Media móvil exponencial
            df_features[f'ewm_mean_{span}'] = df_features[self.target_column].ewm(span=span).mean()
            
            # Desviación estándar exponencial
            df_features[f'ewm_std_{span}'] = df_features[self.target_column].ewm(span=span).std()
            
            # Volatilidad relativa
            df_features[f'ewm_var_coef_{span}'] = (df_features[f'ewm_std_{span}'] / 
                                                  df_features[f'ewm_mean_{span}'])
            
        self.df_features = df_features
        return df_features
    
    def create_diff_features(self, periods=[1, 12], orders=[1, 2]):
        """
        Crea características de diferenciación.
        
        Args:
            periods: Lista de períodos para diferenciar
            orders: Lista de órdenes de diferenciación
            
        Returns:
            DataFrame con características de diferenciación añadidas
        """
        logger.info(f"Creando características de diferenciación períodos={periods}, órdenes={orders}")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Para cada período
        for period in periods:
            # Diferencias de primer orden
            df_features[f'diff_{period}'] = df_features[self.target_column].diff(period)
            
            # Porcentaje de cambio
            df_features[f'pct_change_{period}'] = df_features[self.target_column].pct_change(period)
            
            # Para órdenes superiores
            if 2 in orders:
                df_features[f'diff2_{period}'] = df_features[f'diff_{period}'].diff(period)
                
        self.df_features = df_features
        return df_features
    
    def create_interaction_features(self):
        """
        Crea características basadas en interacciones entre variables existentes.
        
        Returns:
            DataFrame con características de interacción añadidas
        """
        logger.info("Creando características de interacción")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Lags recientes / media móvil (indica si estamos por encima/debajo de la tendencia)
        if 'lag_1' in df_features.columns and 'rolling_mean_12' in df_features.columns:
            df_features['lag_ratio_trend'] = df_features['lag_1'] / df_features['rolling_mean_12']
            
        # Ratio corto/largo plazo
        if 'rolling_mean_3' in df_features.columns and 'rolling_mean_12' in df_features.columns:
            df_features['ratio_trend_3_12'] = df_features['rolling_mean_3'] / df_features['rolling_mean_12']
            
        # Ratio mínimo/máximo (12 meses)
        if 'rolling_min_12' in df_features.columns and 'rolling_max_12' in df_features.columns:
            df_features['ratio_min_max_12'] = df_features['rolling_min_12'] / df_features['rolling_max_12']
            
        # Índice estacional: valor actual / media móvil
        if 'rolling_mean_12' in df_features.columns:
            df_features['seasonal_index'] = df_features[self.target_column] / df_features['rolling_mean_12']
            
        # Amplitud estacional: max - min
        if 'rolling_max_12' in df_features.columns and 'rolling_min_12' in df_features.columns:
            df_features['seasonal_amplitude'] = df_features['rolling_max_12'] - df_features['rolling_min_12']
            
        # Volatilidad relativa
        if 'rolling_std_12' in df_features.columns and 'rolling_mean_12' in df_features.columns:
            df_features['volatility_index'] = df_features['rolling_std_12'] / df_features['rolling_mean_12']
            
        # Características de momento
        if 'lag_1' in df_features.columns and 'lag_12' in df_features.columns:
            df_features['momentum_1_12'] = df_features['lag_1'] - df_features['lag_12']
            
        # Si hay información de precio o ventas totales
        if 'ventas_totales' in self.df.columns:
            # Precio medio
            df_features['precio_medio'] = self.df['ventas_totales'] / self.df[self.target_column]
            
            # Lags de precio
            df_features['precio_medio_lag_1'] = df_features['precio_medio'].shift(1)
            df_features['precio_medio_lag_3'] = df_features['precio_medio'].shift(3)
            
            # Ratio de precios
            df_features['precio_ratio_1_3'] = df_features['precio_medio_lag_1'] / df_features['precio_medio_lag_3']
            
            # Precio normalizado
            df_features['precio_norm'] = df_features['precio_medio'] / df_features['precio_medio'].rolling(window=12).mean()
            
        self.df_features = df_features
        return df_features
    
    def create_date_features(self):
        """
        Crea características avanzadas basadas en fecha y calendario.
        
        Returns:
            DataFrame con características de fecha añadidas
        """
        logger.info("Creando características avanzadas de fecha")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Detectar automáticamente temporadas basado en medias mensuales
        monthly_avg = df_features.groupby('mes')[self.target_column].mean()
        global_avg = monthly_avg.mean()
        
        # Meses de temporada alta (>= media)
        high_season_months = monthly_avg[monthly_avg >= global_avg].index.tolist()
        df_features['temporada_alta'] = df_features['mes'].isin(high_season_months).astype(int)
        
        # Distancia al mes pico
        peak_month = monthly_avg.idxmax()
        df_features['dist_mes_pico'] = (df_features['mes'] - peak_month) % 12
        df_features['dist_mes_pico'] = df_features['dist_mes_pico'].apply(lambda x: min(x, 12 - x))
        
        # Días desde inicio de año
        df_features['dias_desde_inicio_año'] = df_features['día_año'] - 1
        
        # Días hasta fin de año
        df_features['dias_hasta_fin_año'] = 365 - df_features['día_año']
        
        # Días desde/hasta inicio/fin de trimestre
        quarter_days = {1: 31+28+31, 2: 30+31+30, 3: 31+31+30, 4: 31+30+31}
        quarter_starts = {1: 1, 2: 31+28+31+1, 3: 31+28+31+30+31+30+1, 4: 31+28+31+30+31+30+31+31+30+1}
        
        df_features['dias_desde_inicio_trimestre'] = df_features.apply(
            lambda x: x['día_año'] - quarter_starts[x['trimestre']], axis=1)
        
        df_features['dias_hasta_fin_trimestre'] = df_features.apply(
            lambda x: quarter_days[x['trimestre']] - x['dias_desde_inicio_trimestre'], axis=1)
        
        self.df_features = df_features
        return df_features
    
    def create_trend_features(self):
        """
        Crea características de tendencia.
        
        Returns:
            DataFrame con características de tendencia añadidas
        """
        logger.info("Creando características de tendencia")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Índice numérico
        df_features['trend_idx'] = np.arange(len(df_features))
        
        # Logaritmo del índice (tendencia no-lineal)
        df_features['trend_log'] = np.log1p(df_features['trend_idx'])
        
        # Raíz cuadrada (otra no-linealidad)
        df_features['trend_sqrt'] = np.sqrt(df_features['trend_idx'])
        
        # Tendencia cuadrática y cúbica
        df_features['trend_squared'] = df_features['trend_idx'] ** 2
        df_features['trend_cubed'] = df_features['trend_idx'] ** 3
        
        # Tendencia global/local
        overall_trend = np.polyfit(np.arange(len(df_features)), df_features[self.target_column], 1)[0]
        df_features['global_trend'] = overall_trend
        
        # Tendencia local (últimos 12 puntos, o menos si no hay suficientes)
        df_features['local_trend'] = df_features[self.target_column].rolling(12, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=True)
        
        # Ratio tendencia local/global
        df_features['trend_ratio'] = df_features['local_trend'] / (df_features['global_trend'] + 1e-10)
        
        self.df_features = df_features
        return df_features
    
    def create_fourier_features(self, periods=[12, 6, 4, 3], orders=[1, 2, 3, 4]):
        """
        Crea características de Fourier para múltiples periodicidades.
        
        Args:
            periods: Lista de periodicidades a modelar
            orders: Lista de órdenes de Fourier a incluir
            
        Returns:
            DataFrame con características de Fourier añadidas
        """
        logger.info(f"Creando características de Fourier períodos={periods}, órdenes={orders}")
        
        if self.df_features is None:
            self.create_time_features()
            
        df_features = self.df_features.copy()
        
        # Índice temporal normalizado (0 a 1 por cada período)
        t = np.arange(len(df_features))
        
        # Para cada período
        for period in periods:
            # Escalamos el tiempo al período
            scaled_t = 2 * np.pi * (t / period)
            
            # Para cada orden
            for order in orders:
                # Términos seno y coseno
                df_features[f'fourier_sin_{period}_{order}'] = np.sin(order * scaled_t)
                df_features[f'fourier_cos_{period}_{order}'] = np.cos(order * scaled_t)
                
        self.df_features = df_features
        return df_features
    
    def create_all_features(self):
        """
        Crea todas las características disponibles.
        
        Returns:
            DataFrame con todas las características
        """
        logger.info("Creando todas las características disponibles")
        
        # 1. Características básicas de tiempo
        self.create_time_features()
        
        # 2. Características cíclicas
        self.create_cyclical_features()
        
        # 3. Lags
        self.create_lag_features(max_lag=13)
        
        # 4. Ventanas móviles
        self.create_rolling_features(windows=[3, 6, 12, 24])
        
        # 5. Medias móviles exponenciales
        self.create_ewm_features(spans=[3, 6, 12, 24])
        
        # 6. Diferenciaciones
        self.create_diff_features(periods=[1, 3, 6, 12], orders=[1, 2])
        
        # 7. Interacciones
        self.create_interaction_features()
        
        # 8. Características de fecha
        self.create_date_features()
        
        # 9. Tendencias
        self.create_trend_features()
        
        # 10. Fourier para múltiples estacionalidades
        self.create_fourier_features()
        
        # Eliminar filas con valores NaN
        self.df_features = self.df_features.dropna()
        
        # Información sobre características creadas
        n_features = self.df_features.shape[1]
        n_rows = self.df_features.shape[0]
        logger.info(f"Creación de características completada: {n_features} características, {n_rows} filas")
        
        # Información detallada de las columnas
        col_info = pd.DataFrame({
            'column': self.df_features.columns,
            'dtype': self.df_features.dtypes,
            'na_count': self.df_features.isna().sum(),
            'na_pct': (self.df_features.isna().sum() / len(self.df_features) * 100)
        })
        
        logger.info(f"Resumen de características:\n{col_info}")
        
        # Guardar información en archivo
        col_info.to_csv(os.path.join(RESULTS_DIR, "feature_summary.csv"), index=False)
        
        return self.df_features
    
    def plot_feature_distributions(self, top_n=20):
        """
        Genera gráficos de distribución para las características más importantes.
        
        Args:
            top_n: Número de características a mostrar
        """
        if self.feature_importances is None or top_n <= 0:
            return
            
        # Seleccionar top_n características
        top_features = self.feature_importances.head(top_n)['feature'].tolist()
        
        # Crear gráficos
        n_cols = 2
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                if feature in self.df_features.columns:
                    sns.histplot(self.df_features[feature], kde=True, ax=axes[i])
                    axes[i].set_title(f'{feature}')
                    axes[i].set_xlabel('')
        
        # Ocultar ejes vacíos
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"))
        plt.close()
        
    def plot_feature_correlations(self, n_features=25):
        """
        Genera una matriz de correlación para las características top.
        
        Args:
            n_features: Número de características a incluir
        """
        if self.feature_importances is None or n_features <= 0:
            return
            
        # Seleccionar características por importancia
        top_features = self.feature_importances.head(n_features)['feature'].tolist()
        
        # Asegurar que el target esté incluido
        if self.target_column not in top_features:
            top_features = [self.target_column] + top_features[:n_features-1]
            
        # Calcular matriz de correlación
        corr_matrix = self.df_features[top_features].corr()
        
        # Graficar mapa de calor
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5)
        
        plt.title("Matriz de Correlación de Características", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_correlations.png"))
        plt.close()
        
    def select_features(self, model=None, threshold=FEATURE_SELECTION_THRESHOLD):
        """
        Selecciona las características más relevantes según un modelo.
        
        Args:
            model: Modelo a usar para la selección (debe tener feature_importances_)
            threshold: Umbral de importancia para selección
            
        Returns:
            Lista de nombres de características seleccionadas
        """
        logger.info(f"Seleccionando características (umbral={threshold})")
        
        if model is None or not hasattr(model, 'feature_importances_'):
            logger.warning("Modelo no proporcionado o no tiene feature_importances_")
            return list(self.df_features.columns)
            
        # Crear DataFrame con importancias
        feature_names = self.df_features.columns
        importances = model.feature_importances_
        
        self.feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Guardar en archivo
        self.feature_importances.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)
        
        # Graficar top 30 características
        top_n = min(30, len(feature_names))
        plt.figure(figsize=(12, top_n * 0.4))
        sns.barplot(x='importance', y='feature', data=self.feature_importances.head(top_n))
        plt.title('Importancia de Características', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importances.png"))
        plt.close()
        
        # Seleccionar características según umbral
        selected_features = self.feature_importances[self.feature_importances['importance'] > threshold]['feature'].tolist()
        
        # Asegurar que haya un mínimo de características
        min_features = 5
        if len(selected_features) < min_features:
            selected_features = self.feature_importances.head(min_features)['feature'].tolist()
            
        logger.info(f"Seleccionadas {len(selected_features)} características de {len(feature_names)}")
        logger.info(f"Top 10 características: {', '.join(selected_features[:10])}")
        
        self.selected_features = selected_features
        
        # Generar gráficos adicionales
        self.plot_feature_distributions()
        self.plot_feature_correlations()
        
        return selected_features
    
    def get_feature_data(self, selected_only=False):
        """
        Devuelve el DataFrame con todas las características.
        
        Args:
            selected_only: Si True, devuelve solo las características seleccionadas
            
        Returns:
            DataFrame con características
        """
        if selected_only and self.selected_features is not None:
            # Asegurar que el target esté incluido
            selected_cols = self.selected_features.copy()
            if self.target_column not in selected_cols:
                selected_cols.append(self.target_column)
                
            return self.df_features[selected_cols]
        else:
            return self.df_features
