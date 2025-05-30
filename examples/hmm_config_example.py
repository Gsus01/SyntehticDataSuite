"""
Ejemplo de configuración para el modelo HMM (Hidden Markov Model)

Este archivo muestra las diferentes configuraciones disponibles para entrenar
un modelo HMM usando el SyntheticDataSuite.
"""

# Ejemplo 1: Configuración básica de HMM
hmm_config_basic = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["feature1", "feature2"],  # Columnas a usar para el entrenamiento
    "hmm_n_states": 3,  # Número de estados ocultos (equivalente a n_components en el ejemplo)
    "hmm_covariance_type": "diag",  # Tipo de covarianza
    "hmm_n_iter": 100,  # Número máximo de iteraciones
}

# Ejemplo 1b: Configuración con auto-selección de columnas (NUEVO)
hmm_config_auto = {
    "model_type": "hmm",
    # hmm_columns_to_use no especificado = usa todas las columnas numéricas automáticamente
    "hmm_sequence_column": "timestamp",  # Se excluirá automáticamente de las features
    "hmm_n_states": 3,
    "hmm_covariance_type": "diag",
    "hmm_n_iter": 100,
}

# Ejemplo 2: Configuración avanzada con más parámetros
hmm_config_advanced = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["x", "y", "z"],  # Múltiples features
    "hmm_n_states": 5,  # Más estados ocultos para patrones complejos
    "hmm_covariance_type": "full",  # Para variables correlacionadas
    "hmm_n_iter": 200,  # Más iteraciones para convergencia
    "hmm_random_state": 42,  # Para reproducibilidad
    "hmm_tol": 1e-3,  # Tolerancia para convergencia
    "hmm_algorithm": "viterbi",  # Algoritmo de decodificación
}

# Ejemplo 3: Configuración para datos con secuencia temporal explícita
hmm_config_time_series = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["temperature", "humidity", "pressure"],
    "hmm_sequence_column": "timestamp",  # Columna para ordenar temporalmente los datos
    "hmm_n_states": 4,  # Ej: 4 estaciones/regímenes climáticos
    "hmm_covariance_type": "tied",  # Si los estados tienen la misma variación
    "hmm_n_iter": 150,
    "hmm_random_state": 123,
}

# Ejemplo 4: Configuración restrictiva para datos simples
hmm_config_simple = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["value"],  # Una sola variable
    "hmm_n_states": 2,  # Solo dos estados (ej: alto/bajo)
    "hmm_covariance_type": "spherical",  # Muy restrictivo, útil para variables similares
    "hmm_n_iter": 50,  # Menos iteraciones para datos simples
}

# Explicación de los parámetros:
"""
hmm_columns_to_use: (OPCIONAL) Lista de columnas del DataFrame a usar para entrenar el modelo
- Si no se especifica, usa automáticamente todas las columnas numéricas
- Excluye automáticamente: columna de secuencia, 'id', 'index', 'true_state', 'label', 'target'

hmm_n_states: Número de estados ocultos (clusters/regímenes)
- Más estados = modelo más complejo que puede capturar más patrones
- Menos estados = modelo más simple y generalizable

hmm_covariance_type: Cómo modelar la varianza de las observaciones
- 'diag': Rápido, útil si las variables son independientes (por defecto)
- 'full': Para variables correlacionadas, más flexible pero más lento
- 'tied': Si todos los estados tienen la misma variación
- 'spherical': Muy restrictivo, solo si las variables tienen variación similar

hmm_n_iter: Número máximo de iteraciones para el algoritmo EM

hmm_sequence_column: (Opcional) Columna para ordenar los datos temporalmente
- Se excluye automáticamente de las features si no se especifica hmm_columns_to_use

hmm_random_state: Semilla para reproducibilidad

hmm_tol: Tolerancia para convergencia (por defecto 1e-2)

hmm_algorithm: Algoritmo de decodificación (por defecto 'viterbi')
"""

# Ejemplo de uso en el flujo de entrenamiento:
"""
from src.flows.training import training_flow

# Entrenar modelo HMM
model_path = training_flow(
    data_path="data/my_time_series.csv",
    model_name="weather_hmm_model", 
    config=hmm_config_time_series
)

print(f"Modelo HMM entrenado y guardado en: {model_path}")
"""

# Ejemplo de configuración dinámica (similar al input interactivo original):
def create_interactive_hmm_config(columns_to_use, n_states=None, covariance_type=None, n_iter=None):
    """
    Función para crear configuración HMM de manera interactiva
    (equivalente a los inputs del ejemplo original pero programático)
    """
    config = {
        "model_type": "hmm",
        "hmm_columns_to_use": columns_to_use,
    }
    
    # Valores por defecto o personalizados
    config["hmm_n_states"] = n_states or 3
    config["hmm_covariance_type"] = covariance_type or "diag"  
    config["hmm_n_iter"] = n_iter or 100
    config["hmm_random_state"] = 42
    
    return config

# Ejemplo de uso:
"""
# Configuración dinámica
dynamic_config = create_interactive_hmm_config(
    columns_to_use=["feature1", "feature2"],
    n_states=4,
    covariance_type="full",
    n_iter=150
)
""" 