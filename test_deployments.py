#!/usr/bin/env python3
"""
Script de prueba para validar el funcionamiento completo del sistema de deployments.
"""

from src.flows.main import entry_point_flow

def test_complete_pipeline():
    """
    Prueba completa del pipeline de deployments.
    """
    print('🧪 Probando el sistema completo de deployments...')
    print()

    try:
        # Test 1: Entrenar un modelo HMM
        print('1️⃣ Entrenando modelo HMM...')
        training_result = entry_point_flow(
            flow_type='training',
            model_config_name='hmm_default',
            model_name='hmm_test_deployment'
        )
        print(f'   ✅ Resultado: {training_result}')
        print()

        # Test 2: Generar datos con el modelo entrenado
        print('2️⃣ Generando datos sintéticos...')
        generation_result = entry_point_flow(
            flow_type='generation',
            model_name='hmm_test_deployment',
            n_samples=300
        )
        print(f'   ✅ Resultado: {generation_result}')
        print()

        print('🎉 ¡Todos los tests pasaron exitosamente!')
        return True
        
    except Exception as e:
        print(f'❌ Error en las pruebas: {str(e)}')
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    exit(0 if success else 1) 