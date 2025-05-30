"""
Script de ejemplo para probar el modelo HMM con datos sintéticos.

Este script:
1. Genera datos de ejemplo simulando un sistema con estados ocultos
2. Entrena un modelo HMM
3. Genera datos sintéticos
4. Predice estados ocultos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.flows.training import training_flow
from src.models.hmm_model import HMMWrapper

def generate_sample_data(n_samples=500, save_path="data/sample_hmm_data.csv"):
    """
    Genera datos de ejemplo que simulan un sistema con 3 estados ocultos.
    
    Ejemplo: datos de un sensor con 3 regímenes de operación:
    - Estado 0: Normal (baja temperatura, baja vibración)
    - Estado 1: Carga media (temperatura media, vibración media)  
    - Estado 2: Sobrecarga (alta temperatura, alta vibración)
    """
    np.random.seed(42)
    
    # Definir los parámetros de cada estado
    states_params = {
        0: {"temp_mean": 25, "temp_std": 2, "vibr_mean": 10, "vibr_std": 1},  # Normal
        1: {"temp_mean": 45, "temp_std": 3, "vibr_mean": 25, "vibr_std": 2},  # Media carga
        2: {"temp_mean": 70, "temp_std": 4, "vibr_mean": 45, "vibr_std": 3},  # Sobrecarga
    }
    
    # Matriz de transición (probabilidades de cambio entre estados)
    transition_matrix = np.array([
        [0.7, 0.25, 0.05],  # Desde estado 0
        [0.3, 0.6, 0.1],    # Desde estado 1  
        [0.1, 0.4, 0.5]     # Desde estado 2
    ])
    
    # Generar secuencia de estados ocultos
    states = [0]  # Comenzar en estado normal
    for _ in range(n_samples - 1):
        current_state = states[-1]
        next_state = np.random.choice(3, p=transition_matrix[current_state])
        states.append(next_state)
    
    # Generar observaciones basadas en los estados
    temperatures = []
    vibrations = []
    
    for state in states:
        params = states_params[state]
        
        temp = np.random.normal(params["temp_mean"], params["temp_std"])
        vibr = np.random.normal(params["vibr_mean"], params["vibr_std"])
        
        temperatures.append(temp)
        vibrations.append(vibr)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'temperature': temperatures,
        'vibration': vibrations,
        'true_state': states  # Estados reales (en la práctica no los conocemos)
    })
    
    # Guardar datos
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Datos de ejemplo generados: {save_path}")
    print(f"📊 Forma de los datos: {df.shape}")
    print(f"🎯 Estados únicos: {sorted(df['true_state'].unique())}")
    print(f"📈 Estadísticas por estado:")
    print(df.groupby('true_state')[['temperature', 'vibration']].mean().round(2))
    
    return df

def test_hmm_training():
    """
    Prueba el entrenamiento del modelo HMM.
    """
    print("\n" + "="*50)
    print("🏋️ PROBANDO ENTRENAMIENTO DE MODELO HMM")
    print("="*50)
    
    # Configuración del modelo HMM
    hmm_config = {
        "model_type": "hmm",
        "hmm_columns_to_use": ["temperature", "vibration"],
        "hmm_sequence_column": "timestamp",  # Para ordenar temporalmente
        "hmm_n_states": 3,
        "hmm_covariance_type": "diag",
        "hmm_n_iter": 100,
        "hmm_random_state": 42,
    }
    
    try:
        # Entrenar modelo
        model_path = training_flow(
            data_path="data/sample_hmm_data.csv",
            model_name="sensor_hmm_test",
            config=hmm_config
        )
        
        print(f"✅ Modelo entrenado exitosamente")
        print(f"📦 Guardado en: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return None

def test_hmm_training_auto():
    """
    Prueba el entrenamiento del modelo HMM con auto-selección de columnas.
    """
    print("\n" + "="*50)
    print("🏋️ PROBANDO ENTRENAMIENTO CON AUTO-SELECCIÓN DE COLUMNAS")
    print("="*50)
    
    # Configuración del modelo HMM sin especificar columnas
    hmm_config_auto = {
        "model_type": "hmm",
        # hmm_columns_to_use no especificado = auto-selección
        "hmm_sequence_column": "timestamp",  # Se excluirá automáticamente
        "hmm_n_states": 3,
        "hmm_covariance_type": "diag", 
        "hmm_n_iter": 100,
        "hmm_random_state": 42,
    }
    
    try:
        # Entrenar modelo con auto-selección
        model_path = training_flow(
            data_path="data/sample_hmm_data.csv",
            model_name="sensor_hmm_auto_test",
            config=hmm_config_auto
        )
        
        print(f"✅ Modelo con auto-selección entrenado exitosamente")
        print(f"📦 Guardado en: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error en entrenamiento con auto-selección: {e}")
        return None

def test_hmm_sampling(model_path):
    """
    Prueba la generación de datos sintéticos.
    """
    print("\n" + "="*50)
    print("🧠 PROBANDO GENERACIÓN DE DATOS SINTÉTICOS")
    print("="*50)
    
    try:
        # Cargar modelo
        hmm_model = HMMWrapper(
            model_name="sensor_hmm_test",
            model_path=model_path,
            config={}
        )
        hmm_model.load_model()
        
        # Generar datos sintéticos
        n_synthetic = 100
        synthetic_data = hmm_model.sample(n_samples=n_synthetic)
        
        print(f"✅ Generados {n_synthetic} puntos sintéticos")
        print(f"📊 Columnas: {list(synthetic_data.columns)}")
        print(f"🎯 Estados únicos generados: {sorted(synthetic_data['hidden_state'].unique())}")
        
        print("\n📈 Primeras 10 muestras:")
        print(synthetic_data.head(10)[['temperature', 'vibration', 'hidden_state']].round(2))
        
        print("\n📊 Estadísticas por estado generado:")
        stats = synthetic_data.groupby('hidden_state')[['temperature', 'vibration']].agg(['mean', 'std']).round(2)
        print(stats)
        
        return synthetic_data
        
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        return None

def test_hmm_prediction(model_path):
    """
    Prueba la predicción de estados ocultos.
    """
    print("\n" + "="*50)
    print("🔮 PROBANDO PREDICCIÓN DE ESTADOS OCULTOS")
    print("="*50)
    
    try:
        # Cargar modelo
        hmm_model = HMMWrapper(
            model_name="sensor_hmm_test",
            model_path=model_path,
            config={}
        )
        hmm_model.load_model()
        
        # Crear datos de prueba representativos de cada estado
        test_data = pd.DataFrame({
            'temperature': [25, 45, 70, 30, 50],  # Diferentes regímenes
            'vibration': [10, 25, 45, 12, 30]
        })
        
        # Predecir estados
        predicted_states = hmm_model.predict_hidden_states(test_data)
        
        print("✅ Predicción completada")
        print("\n📊 Datos de entrada vs Estados predichos:")
        
        test_with_predictions = test_data.copy()
        test_with_predictions['predicted_state'] = predicted_states
        print(test_with_predictions)
        
        # Interpretación esperada (aproximada)
        print("\n💡 Interpretación esperada:")
        print("Estado 0: Normal (temp ~25, vibr ~10)")
        print("Estado 1: Media carga (temp ~45, vibr ~25)")  
        print("Estado 2: Sobrecarga (temp ~70, vibr ~45)")
        
        return predicted_states
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return None

def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    print("🚀 INICIANDO PRUEBAS DEL MODELO HMM")
    print("="*50)
    
    # 1. Generar datos de ejemplo
    print("1️⃣ Generando datos de ejemplo...")
    sample_data = generate_sample_data()
    
    if sample_data is not None:
        # 2. Entrenar modelo
        print("\n2️⃣ Entrenando modelo HMM...")
        model_path = test_hmm_training()
        
        # 2b. Entrenar modelo con auto-selección
        print("\n2️⃣b Entrenando modelo HMM con auto-selección...")
        model_path_auto = test_hmm_training_auto()
        
        if model_path:
            # 3. Generar datos sintéticos
            print("\n3️⃣ Generando datos sintéticos...")
            synthetic_data = test_hmm_sampling(model_path)
            
            # 4. Predecir estados
            print("\n4️⃣ Prediciendo estados ocultos...")
            predictions = test_hmm_prediction(model_path)
            
            if all([synthetic_data is not None, predictions is not None]):
                print("\n" + "="*50)
                print("🎉 ¡TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE!")
                print("="*50)
                print("\n📋 Resumen:")
                print(f"- Datos originales: {len(sample_data)} puntos")
                if synthetic_data is not None:
                    print(f"- Datos sintéticos: {len(synthetic_data)} puntos")
                if predictions is not None:
                    print(f"- Predicciones: {len(predictions)} estados")
                print(f"- Modelo manual guardado en: {model_path}")
                if model_path_auto:
                    print(f"- Modelo auto-selección guardado en: {model_path_auto}")
            else:
                print("\n⚠️ Algunas pruebas fallaron. Revisa los logs arriba.")
        else:
            print("\n❌ No se pudo completar el entrenamiento.")
    else:
        print("\n❌ No se pudieron generar los datos de ejemplo.")

if __name__ == "__main__":
    main() 