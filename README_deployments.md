# Deployments en Python para Prefect 3 - SyntehticDataSuite

Este documento explica cómo usar los deployments definidos en Python para ejecutar pipelines de entrenamiento y generación de modelos sintéticos.

## 📁 Archivos de Deployment

### 1. `deploy.py` - Deployments Completos
Contiene 8 deployments predefinidos que cubren diferentes escenarios:

- **Training Deployments:**
  - `py-train-gmm-default-csv`: Entrena GMM por defecto
  - `py-train-gmm-experiment-csv`: Entrena GMM experimental (3 componentes)
  - `py-train-hmm-default-csv`: Entrena HMM por defecto
  - `py-multi-model-training`: Pipeline para modelos HMM avanzados

- **Generation Deployments:**
  - `py-generate-gmm-default-csv`: Genera 1500 muestras con GMM
  - `py-generate-hmm-default`: Genera 800 muestras con HMM

- **Specialized Deployments:**
  - `py-training-only-flow`: Flujo dedicado solo para entrenamiento
  - `py-generation-only-flow`: Flujo dedicado solo para generación

### 2. `deploy_simple.py` - Deployments Básicos
Versión simplificada con solo 2 deployments esenciales:
- `simple-training-deployment`: Entrenamiento básico
- `simple-generation-deployment`: Generación básica

## 🚀 Cómo Usar los Deployments

### Opción A: Ejecutar desde Python

```bash
# Deployments completos (8 deployments)
python3 deploy.py

# Deployments simples (2 deployments)
python3 deploy_simple.py
```

### Opción B: Ejecutar deployments específicos desde CLI

```bash
# Entrenamiento
prefect deployment run 'entry-point-flow/py-train-gmm-default-csv'
prefect deployment run 'entry-point-flow/py-train-hmm-default-csv'

# Generación
prefect deployment run 'entry-point-flow/py-generate-gmm-default-csv'
prefect deployment run 'entry-point-flow/py-generate-hmm-default'

# Con parámetros personalizados
prefect deployment run 'entry-point-flow/py-generate-gmm-default-csv' --param n_samples=2000
```

### Opción C: Ejecutar directamente desde Python

```python
from src.flows.main import entry_point_flow

# Entrenamiento
result = entry_point_flow(
    flow_type="training",
    model_config_name="gmm_default",
    input_data_filename="train_FD001.csv"
)

# Generación
result = entry_point_flow(
    flow_type="generation",
    model_config_name="gmm_default",
    n_samples=1500
)
```

## ⚙️ Configuración

### Parámetros Disponibles

#### Para `flow_type="training"`:
- `model_config_name`: Configuración del modelo (ej: `gmm_default`, `hmm_default`)
- `model_name`: Nombre del modelo entrenado (opcional)
- `input_data_filename`: Archivo de datos de entrada (ej: `train_FD001.csv`)

#### Para `flow_type="generation"`:
- `model_config_name`: Configuración del modelo a usar
- `model_name`: Nombre del modelo entrenado a cargar
- `n_samples`: Número de muestras a generar
- `output_path`: Ruta de salida (opcional)

### Configuraciones de Modelo Disponibles

Definidas en `config/config.yaml`:

```yaml
models:
  gmm_default:
    model_type: "gmm"
    n_components: 1
    
  gmm_experiment_01:
    model_type: "gmm"
    n_components: 3
    
  hmm_default:
    model_type: "hmm"
    n_components: 3
    
  hmm_experiment_01:
    model_type: "hmm"
    n_components: 5
```

## 📊 Monitoreo y UI

### Prefect UI
- **URL**: http://127.0.0.1:4200
- **Deployments**: http://127.0.0.1:4200/deployments
- **Flow Runs**: http://127.0.0.1:4200/flow-runs

### Verificar Resultados

```bash
# Ver modelos entrenados
ls -la data/models/

# Ver datos generados
ls -la data/synthetic_output/

# Ver logs de ejecución
tail -f ~/.prefect/logs/prefect.log
```

## 🔍 Ejemplos Prácticos

### 1. Pipeline Completo: Entrenar y Generar

```bash
# 1. Entrenar modelo GMM
prefect deployment run 'entry-point-flow/py-train-gmm-default-csv'

# 2. Generar datos sintéticos
prefect deployment run 'entry-point-flow/py-generate-gmm-default-csv'
```

### 2. Experimentación con Diferentes Modelos

```bash
# Entrenar diferentes modelos
prefect deployment run 'entry-point-flow/py-train-gmm-experiment-csv'
prefect deployment run 'entry-point-flow/py-train-hmm-default-csv'

# Generar con diferentes cantidades
prefect deployment run 'entry-point-flow/py-generate-gmm-default-csv' --param n_samples=500
prefect deployment run 'entry-point-flow/py-generate-hmm-default' --param n_samples=1200
```

### 3. Automatización con Scripts

```python
# pipeline_automation.py
from src.flows.main import entry_point_flow

def run_full_pipeline():
    # Entrenar
    training_result = entry_point_flow(
        "training", 
        model_config_name="gmm_experiment_01",
        model_name="my_experimental_model"
    )
    
    # Generar
    generation_result = entry_point_flow(
        "generation",
        model_name="my_experimental_model",
        n_samples=2000
    )
    
    return training_result, generation_result
```

## 🛠️ Solución de Problemas

### Errores Comunes

1. **Modelo no encontrado**: Asegúrate de entrenar el modelo antes de generar
2. **Archivo de datos no encontrado**: Verifica que `data/train_FD001.csv` existe
3. **Configuración no encontrada**: Revisa que `config/config.yaml` contiene la configuración del modelo

### Debug

```bash
# Ver estado de deployments
prefect deployment ls

# Ver logs detallados
prefect config set PREFECT_LOGGING_LEVEL=DEBUG

# Ejecutar flow directamente para debug
python3 -c "
from src.flows.main import entry_point_flow
result = entry_point_flow('training', model_config_name='gmm_default')
print(f'Result: {result}')
"
```

## 🚀 Ventajas de Deployments en Python vs YAML

### Prefect 3 con Python:
- ✅ **Flexibilidad**: Lógica programática para crear deployments
- ✅ **Tipado**: Type hints y validación en tiempo de desarrollo
- ✅ **Reutilización**: Funciones para crear deployments similares
- ✅ **Integración**: Acceso directo a configuración YAML y flujos

### Comparación con prefect.yaml:
- ✅ **Python**: Más poderoso para casos complejos
- ✅ **YAML**: Más simple para casos básicos
- ✅ **Ambos**: Válidos en Prefect 3

Este sistema de deployments en Python te permite tener control total sobre tus pipelines de ML mientras mantienes la configuración centralizada en YAML. 