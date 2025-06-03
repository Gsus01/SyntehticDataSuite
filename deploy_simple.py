# deploy_simple.py - Definición simple de deployments para Prefect 3
"""
Este archivo proporciona opciones para testing rápido y deployments simples con Prefect 3.
Incluye modo de testing rápido sin deployments y modo serve para deployments locales.
"""

import argparse
import sys
from src.flows.main import entry_point_flow, training_only_flow, generation_only_flow

def create_training_deployment():
    """
    Crea un deployment para entrenamiento de modelos usando serve.
    """
    print("🏋️ Creando deployment de entrenamiento...")
    
    deployment = entry_point_flow.serve(
        name="simple-training-deployment",
        description="Deployment simple para entrenar modelos usando configuración YAML",
        parameters={
            "flow_type": "training",
            "model_config_name": "hmm_default",
            "input_data_filename": "train_FD001.csv"
        },
        tags=["training", "simple", "python-defined"],
    )
    
    print("✅ Training deployment creado:")
    print(f"   - Nombre: simple-training-deployment")
    print(f"   - Comando: prefect deployment run 'entry-point-flow/simple-training-deployment'")
    
    return deployment

def create_generation_deployment():
    """
    Crea un deployment para generación de datos sintéticos usando serve.
    """
    print("\n🧠 Creando deployment de generación...")
    
    deployment = entry_point_flow.serve(
        name="simple-generation-deployment",
        description="Deployment simple para generar datos sintéticos usando modelos entrenados",
        parameters={
            "flow_type": "generation",
            "model_config_name": "hmm_default",
            "n_samples": 1000
        },
        tags=["generation", "simple", "python-defined"],
    )
    
    print("✅ Generation deployment creado:")
    print(f"   - Nombre: simple-generation-deployment")
    print(f"   - Comando: prefect deployment run 'entry-point-flow/simple-generation-deployment'")
    
    return deployment

def run_quick_test(test_type="both"):
    """
    Ejecuta tests rápidos directamente sin deployments.
    Perfecto para pruebas rápidas durante desarrollo.
    """
    print("🚀 Ejecutando tests rápidos...")
    
    try:
        if test_type in ["training", "both"]:
            print("\n🏋️ Ejecutando flow de entrenamiento...")
            result = entry_point_flow(
                flow_type="training",
                model_config_name="gmm_default",
                input_data_filename="train_FD001.csv"
            )
            print(f"✅ Entrenamiento completado: {result}")
        
        if test_type in ["generation", "both"]:
            print("\n🧠 Ejecutando flow de generación...")
            result = entry_point_flow(
                flow_type="generation",
                model_config_name="gmm_default",
                n_samples=100  # Número pequeño para test rápido
            )
            print(f"✅ Generación completada: {result}")
            
        print("\n✅ Tests rápidos completados exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error durante tests rápidos: {str(e)}")
        return False

def show_usage_examples():
    """
    Muestra ejemplos de cómo usar los tests y deployments.
    """
    print("\n📚 Ejemplos de uso:")
    
    print("\n🔥 TESTING RÁPIDO (recomendado para desarrollo):")
    print("   python deploy_simple.py --quick-test")
    print("   python deploy_simple.py --quick-test --type training")
    print("   python deploy_simple.py --quick-test --type generation")
    
    print("\n⚡ DEPLOYMENTS CON PROCESO SIRVIENDO:")
    print("   python deploy_simple.py --serve")
    print("   # Mantiene terminal abierta sirviendo deployments")
    
    print("\n💻 Ejecutar deployments desde línea de comandos:")
    print("   prefect deployment run 'entry-point-flow/simple-training-deployment'")
    print("   prefect deployment run 'entry-point-flow/simple-generation-deployment'")
    
    print("\n🎯 Ejecutar con parámetros personalizados:")
    print("   prefect deployment run 'entry-point-flow/simple-training-deployment' --param model_config_name=gmm_experiment_01")
    print("   prefect deployment run 'entry-point-flow/simple-generation-deployment' --param n_samples=2000")
    
    print("\n🌐 Ver deployments en la UI de Prefect:")
    print("   http://127.0.0.1:4200/deployments")

def main():
    """
    Función principal con opciones simplificadas de ejecución.
    """
    parser = argparse.ArgumentParser(description="Sistema de testing y deployments simples para Prefect 3")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Ejecutar tests rápidos sin deployments (RECOMENDADO para desarrollo)")
    parser.add_argument("--type", choices=["training", "generation", "both"], 
                       default="both", help="Tipo de test a ejecutar")
    parser.add_argument("--serve", action="store_true",
                       help="Crear deployments con proceso sirviendo")
    parser.add_argument("--help-examples", action="store_true",
                       help="Mostrar ejemplos de uso")
    
    args = parser.parse_args()
    
    if args.help_examples:
        show_usage_examples()
        return True
    
    if args.quick_test:
        print("🚀 Modo TEST RÁPIDO - Ejecutando flows directamente...")
        return run_quick_test(args.type)
    
    # El modo por defecto es serve si no se especifica quick-test
    if not args.serve:
        args.serve = True
        
    if args.serve:
        print("🚀 Creando deployments con proceso sirviendo...")
        print("💡 Esta versión mantiene la terminal abierta sirviendo deployments.")
    
    try:
        # Crear deployments usando serve
        training_deployment = create_training_deployment()
        generation_deployment = create_generation_deployment()
        
        print(f"\n📊 Resumen:")
        print(f"✅ 2 deployments creados exitosamente en modo serve")
        
        show_usage_examples()
        print("\n✅ Deployments sirviendo!")
        print("💡 Presiona Ctrl+C para detener el servidor cuando termines.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al crear deployments: {str(e)}")
        return False

if __name__ == "__main__":
    # Si no se pasan argumentos, mostrar ayuda rápida
    if len(sys.argv) == 1:
        print("🚀 Deploy Simple para Prefect 3")
        print("\n💡 Para TESTING RÁPIDO (recomendado):")
        print("   python deploy_simple.py --quick-test")
        print("\n💡 Para DEPLOYMENTS:")
        print("   python deploy_simple.py --serve")
        print("\n💡 Para ver todas las opciones:")
        print("   python deploy_simple.py --help")
        print("   python deploy_simple.py --help-examples")
        sys.exit(0)
    
    success = main()
    if not success:
        print("\n❌ Falló la operación.")
        sys.exit(1) 