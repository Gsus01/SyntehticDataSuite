# Modelo HMM (Hidden Markov Model) - Guía de Uso

## Descripción

El **Hidden Markov Model (HMM)** es un modelo estadístico que supone que el sistema que se está modelando es un proceso de Markov con estados no observados (ocultos). Es especialmente útil para:

- **Series temporales** con regímenes o estados subyacentes
- **Datos secuenciales** donde el orden importa
- **Detección de patrones** en comportamientos cambiantes
- **Segmentación** de secuencias en diferentes estados

## Características del HMM en SyntheticDataSuite

### Funcionalidades principales:
- ✅ Entrenamiento con datos multivariados
- ✅ Generación de datos sintéticos secuenciales
- ✅ Predicción de estados ocultos
- ✅ Manejo de columnas de secuencia temporal
- ✅ Configuración flexible de hiperparámetros
- ✅ Guardado y carga de modelos entrenados

### Funcionalidades adicionales:
- **Estados ocultos incluidos**: Los datos generados incluyen la secuencia de estados ocultos
- **Índice secuencial**: Se mantiene el orden temporal en los datos sintéticos
- **Metadatos preservados**: Información sobre columnas y configuración se guarda con el modelo

## Configuración

### Parámetros principales:

```python
config = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["feature1", "feature2"],  # OPCIONAL: si no se especifica, usa todas las columnas numéricas
    "hmm_n_states": 3,                              # Número de estados ocultos
    "hmm_covariance_type": "diag",                  # Tipo de covarianza
    "hmm_n_iter": 100,                              # Máximo de iteraciones
    "hmm_sequence_column": "timestamp",             # OPCIONAL: para ordenar datos
    "hmm_random_state": 42,                         # Para reproducibilidad
    "hmm_tol": 1e-2,                                # Tolerancia de convergencia
    "hmm_algorithm": "viterbi",                     # Algoritmo de decodificación
}
```

### Tipos de covarianza explicados:

| Tipo | Descripción | Cuándo usar |
|------|-------------|-------------|
| `"diag"` | Variables independientes | **Por defecto** - Más rápido, buena opción inicial |
| `"full"` | Variables correlacionadas | Cuando las features están relacionadas |
| `"tied"` | Misma variación en todos los estados | Estados con variabilidad similar |
| `"spherical"` | Variación uniforme | Variables con escalas similares (restrictivo) |

### Auto-selección de columnas:

Si no especificas `hmm_columns_to_use`, el modelo automáticamente:

1. **Selecciona todas las columnas numéricas** del DataFrame
2. **Excluye la columna de secuencia** (si está especificada en `hmm_sequence_column`)
3. **Excluye columnas comunes que no son features**: como 'id', 'index', 'true_state', 'label', 'target'

```python
# Configuración automática - usa todas las columnas numéricas
config_auto = {
    "model_type": "hmm",
    # hmm_columns_to_use no especificado = auto-selección
    "hmm_sequence_column": "timestamp",  # Se excluirá automáticamente
    "hmm_n_states": 3,
}

# Equivale a especificar manualmente todas las columnas numéricas
# (excluyendo 'timestamp' y columnas como 'id', 'true_state', etc.)
```

## Ejemplos de uso

### 1. Caso básico - Series temporales simples

```python
from src.flows.training import training_flow

# Configuración simple con auto-selección de columnas
config = {
    "model_type": "hmm",
    # hmm_columns_to_use no especificado = usa todas las columnas numéricas automáticamente
    "hmm_sequence_column": "timestamp",  # Columna temporal (se excluirá de features)
    "hmm_n_states": 3,  # Ej: frío, templado, calor
    "hmm_covariance_type": "diag",
    "hmm_n_iter": 100,
}

# Entrenar modelo - usará automáticamente columnas como 'temperature', 'humidity', etc.
model_path = training_flow(
    data_path="data/weather_data.csv",  # Debe tener columnas numéricas
    model_name="weather_hmm",
    config=config
)
```

### 2. Caso avanzado - Datos financieros

```python
# Configuración para datos financieros (bull/bear markets)
financial_config = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["returns", "volatility", "volume"],
    "hmm_sequence_column": "date",  # Ordenar por fecha
    "hmm_n_states": 2,  # Bull vs Bear market
    "hmm_covariance_type": "full",  # Variables correlacionadas
    "hmm_n_iter": 200,
    "hmm_random_state": 42,
}

model_path = training_flow(
    data_path="data/stock_data.csv",
    model_name="market_regime_hmm",
    config=financial_config
)
```

### 3. Generación de datos sintéticos

```python
from src.models.hmm_model import HMMWrapper

# Cargar modelo entrenado
hmm_model = HMMWrapper(
    model_name="weather_hmm",
    model_path="./model_registry/weather_hmm/model.pkl",
    config={}  # Se carga del archivo
)
hmm_model.load_model()

# Generar 1000 muestras sintéticas
synthetic_data = hmm_model.sample(n_samples=1000)

print(synthetic_data.head())
# Salida incluye:
# - Las columnas originales (temperature, humidity)
# - hidden_state: estado oculto en cada momento
# - sequence_index: índice temporal
```

### 4. Predicción de estados ocultos

```python
# Predecir estados para nuevos datos
import pandas as pd

new_data = pd.DataFrame({
    'temperature': [25.1, 30.5, 15.2],
    'humidity': [60.2, 45.8, 80.1]
})

hidden_states = hmm_model.predict_hidden_states(new_data)
print(f"Estados predichos: {hidden_states}")
# Salida: [1 2 0] (ejemplo)
```

## Casos de uso recomendados

### 🌟 Ideal para:
- **Series temporales de régimen**: Clima, mercados financieros, comportamiento de usuarios
- **Datos secuenciales**: Logs de eventos, trayectorias, procesos industriales
- **Segmentación temporal**: Identificar períodos con comportamientos distintos
- **Análisis de transiciones**: Estudiar cambios entre estados

### ⚠️ Limitaciones:
- Supone **dependencia de primer orden** (estado actual solo depende del anterior)
- **No adecuado** para dependencias a largo plazo sin estados intermedios
- Requiere que el **orden de los datos sea significativo**

## Comparación con otros modelos

| Característica | HMM | GMM | Ventaja HMM |
|----------------|-----|-----|-------------|
| **Orden temporal** | ✅ Sí | ❌ No | Preserva secuencias |
| **Estados ocultos** | ✅ Sí | ❌ No | Interpretabilidad |
| **Transiciones** | ✅ Modeladas | ❌ No | Dinámicas temporales |
| **Complejidad** | Media | Baja | Balance flexibilidad/simplicidad |

## Troubleshooting

### Problemas comunes:

1. **Modelo no converge**:
   - Aumentar `hmm_n_iter`
   - Reducir `hmm_tol`
   - Cambiar `hmm_covariance_type`

2. **Estados poco interpretables**:
   - Reducir `hmm_n_states`
   - Verificar calidad de los datos
   - Probar diferentes `covariance_type`

3. **Performance lenta**:
   - Usar `"diag"` en lugar de `"full"`
   - Reducir `hmm_n_states`
   - Reducir `hmm_n_iter`

### Validación del modelo:

```python
# Revisar convergencia
print(f"¿Convergió? {hmm_model.model.monitor_.converged}")

# Log-likelihood del modelo
score = hmm_model.model.score(training_data)
print(f"Log-likelihood: {score}")
```

## Referencias

- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- [Hidden Markov Models Tutorial](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
- [Rabiner, L.R. "A tutorial on hidden Markov models"](https://ieeexplore.ieee.org/document/18626) 