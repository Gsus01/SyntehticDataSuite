# SyntheticDataSuite

Suite de herramientas para generar datos sintéticos usando diferentes modelos de machine learning. Diseñado para ser modular, extensible y fácil de usar.

## 🚀 Características

- **Modelos múltiples**: GMM, HMM, y más en desarrollo
- **Configuración flexible**: Sin hardcodeo, todo configurable
- **Flujos automatizados**: Usando Prefect para orquestación
- **Fácil extensión**: Patrón base para agregar nuevos modelos

## 📦 Modelos disponibles

### ✅ Gaussian Mixture Model (GMM)
- Ideal para datos multivariados con clusters
- Generación de datos sintéticos basada en mixturas gaussianas

### ✅ Hidden Markov Model (HMM)
- **NUEVO** - Ideal para series temporales con estados ocultos
- Generación de secuencias sintéticas preservando dinámicas temporales
- Predicción de estados ocultos
- Ver [documentación detallada](examples/README_HMM.md)

## 🛠️ Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd SyntehticDataSuite

# Instalar dependencias
pip install -e .
```

## 🎯 Uso rápido

### Entrenar un modelo HMM:

```python
from src.flows.training import training_flow

# Opción 1: Auto-selección de columnas (recomendado para empezar)
config_auto = {
    "model_type": "hmm",
    # hmm_columns_to_use no especificado = usa todas las columnas numéricas
    "hmm_sequence_column": "timestamp",  # Columna temporal (opcional)
    "hmm_n_states": 3,
    "hmm_covariance_type": "diag",
    "hmm_n_iter": 100,
}

# Opción 2: Especificar columnas manualmente
config_manual = {
    "model_type": "hmm",
    "hmm_columns_to_use": ["temperature", "vibration"],  # Columnas específicas
    "hmm_n_states": 3,
    "hmm_covariance_type": "diag",
    "hmm_n_iter": 100,
}

# Entrenar (funciona con cualquiera de las dos configuraciones)
model_path = training_flow(
    data_path="data/sensor_data.csv",
    model_name="sensor_hmm",
    config=config_auto  # o config_manual
)
```

### Generar datos sintéticos:

```python
from src.models.hmm_model import HMMWrapper

# Cargar modelo
model = HMMWrapper("sensor_hmm", model_path, {})
model.load_model()

# Generar 1000 muestras sintéticas
synthetic_data = model.sample(1000)
```

## 📚 Ejemplos

- **HMM completo**: [`examples/test_hmm_example.py`](examples/test_hmm_example.py)
- **Configuraciones**: [`examples/hmm_config_example.py`](examples/hmm_config_example.py)
- **Documentación HMM**: [`examples/README_HMM.md`](examples/README_HMM.md)

## 🔧 Desarrollo

### Agregar un nuevo modelo:

1. Crear wrapper en `src/models/` heredando de `SyntheticModel`
2. Implementar métodos: `train()`, `sample()`, `save_model()`, `load_model()`
3. Agregar al flujo en `src/flows/training.py`
4. Crear ejemplos y documentación

### Estructura del proyecto:

```
SyntehticDataSuite/
├── src/
│   ├── models/          # Wrappers de modelos
│   │   ├── base.py      # Clase base abstracta
│   │   ├── gmm_model.py # Gaussian Mixture Model
│   │   └── hmm_model.py # Hidden Markov Model
│   └── flows/           # Flujos de Prefect
│       └── training.py  # Flujo de entrenamiento
├── examples/            # Ejemplos y documentación
├── data/               # Datos de ejemplo
└── model_registry/     # Modelos entrenados
```

## 🤝 Contribuir

1. Fork del repositorio
2. Crear rama: `git checkout -b feature/nuevo-modelo`
3. Commit: `git commit -m 'Agregar nuevo modelo'`
4. Push: `git push origin feature/nuevo-modelo`
5. Pull Request

## 📝 Licencia

Ver [LICENSE](LICENSE) para más detalles.