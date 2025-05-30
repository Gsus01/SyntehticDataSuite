#!/usr/bin/env python3
"""
Script para probar el modelo HMM en el flujo principal de SyntheticDataSuite.
"""

import asyncio
from src.flows.main import main_orchestration_flow

def test_hmm_with_main_flow():
    """
    Ejecuta el flujo principal usando el modelo HMM.
    """
    
    # Configuración para usar HMM con columnas específicas de train_FD001.csv
    config = {
        "model_type": "hmm",
        "model_name_prefix": "my_hmm_test",
        "n_samples": 500,  # Número de muestras sintéticas a generar
        
        # Parámetros específicos del HMM
        "hmm_n_states": 3,  # Número de estados ocultos
        "hmm_covariance_type": "diag",  # Tipo de covarianza
        "hmm_n_iter": 100,  # Máximo de iteraciones
        
        # Especificamos columnas específicas del dataset train_FD001.csv
        "hmm_columns_to_use": ["col3", "col4", "col5", "col6", "col7"],  # Primeras 5 columnas numéricas
        
        # Opcional: si tienes una columna de tiempo/secuencia
        # "hmm_sequence_column": "col2",  # col2 parece ser secuencial
    }
    
    # Usar específicamente train_FD001.csv
    raw_data_path = "data/train_FD001.csv"
    
    print("🚀 Iniciando flujo principal con modelo HMM...")
    print(f"📁 Datos: {raw_data_path}")
    print(f"🔧 Configuración: {config}")
    print("-" * 60)
    
    try:
        # Ejecutar el flujo principal
        result = main_orchestration_flow(raw_data_path, config)
        print(f"\n✅ Flujo completado exitosamente!")
        print(f"📈 Datos generados: {result}")
        
    except Exception as e:
        print(f"\n❌ Error en el flujo: {e}")
        raise

if __name__ == "__main__":
    test_hmm_with_main_flow() 