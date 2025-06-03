# deploy.py - Deployments en Python para Prefect 3
from prefect import serve
from src.flows.main import entry_point_flow, training_only_flow, generation_only_flow

def create_deployments():
    """
    Crea y registra deployments usando la nueva API de Prefect 3.
    """
    print("🚀 Creando deployments definidos en Python para Prefect 3...")
    
    # Lista de deployments a crear usando flow.serve()
    deployments = []
    
    # Deployment para entrenamiento con gmm_default
    deployment_train_gmm_default = entry_point_flow.serve(
        name="py-train-gmm-default-csv",
        description="Entrena GMM por defecto usando configuración YAML (definido en Python)",
        parameters={
            "flow_type": "training",
            "model_config_name": "gmm_default",
            "input_data_filename": "train_FD001.csv"
        },
        tags=["training", "gmm", "python-defined", "config-yaml"],
        # work_pool_name="your-agent-pool", # Especifica tu work pool si tienes uno
        # work_queue_name="your-queue" # Opcional
    )
    deployments.append(("GMM Default Training", deployment_train_gmm_default))
    
    # Deployment para entrenamiento con gmm_experiment_01
    deployment_train_gmm_experiment = entry_point_flow.serve(
        name="py-train-gmm-experiment-csv",
        description="Entrena GMM experimental (3 componentes) usando configuración YAML",
        parameters={
            "flow_type": "training",
            "model_config_name": "gmm_experiment_01",
            "model_name": "gmm_exp_model"
        },
        tags=["training", "gmm", "experiment", "python-defined"],
    )
    deployments.append(("GMM Experiment Training", deployment_train_gmm_experiment))
    
    # Deployment para entrenamiento con HMM por defecto
    deployment_train_hmm_default = entry_point_flow.serve(
        name="py-train-hmm-default-csv",
        description="Entrena HMM por defecto usando configuración YAML",
        parameters={
            "flow_type": "training",
            "model_config_name": "hmm_default",
            "input_data_filename": "train_FD001.csv"
        },
        tags=["training", "hmm", "python-defined", "config-yaml"],
    )
    deployments.append(("HMM Default Training", deployment_train_hmm_default))
    
    # Deployment para generación con gmm_default
    deployment_generate_gmm_default = entry_point_flow.serve(
        name="py-generate-gmm-default-csv",
        description="Genera datos con GMM por defecto usando configuración YAML",
        parameters={
            "flow_type": "generation",
            "model_config_name": "gmm_default",
            "n_samples": 1500
        },
        tags=["generation", "gmm", "python-defined", "config-yaml"],
    )
    deployments.append(("GMM Default Generation", deployment_generate_gmm_default))
    
    # Deployment para generación con HMM
    deployment_generate_hmm = entry_point_flow.serve(
        name="py-generate-hmm-default",
        description="Genera datos secuenciales con HMM usando configuración YAML",
        parameters={
            "flow_type": "generation",
            "model_config_name": "hmm_default",
            "model_name": "hmm_default",  # Usar modelo entrenado con este nombre
            "n_samples": 800
        },
        tags=["generation", "hmm", "sequential", "python-defined"],
    )
    deployments.append(("HMM Generation", deployment_generate_hmm))
    
    # Deployments específicos usando flujos dedicados
    deployment_training_only = training_only_flow.serve(
        name="py-training-only-flow",
        description="Flujo dedicado solo para entrenamiento",
        parameters={
            "model_config_name": "gmm_default"
        },
        tags=["training", "dedicated", "python-defined"],
    )
    deployments.append(("Training Only Flow", deployment_training_only))
    
    deployment_generation_only = generation_only_flow.serve(
        name="py-generation-only-flow",
        description="Flujo dedicado solo para generación",
        parameters={
            "model_config_name": "gmm_default",
            "n_samples": 2000
        },
        tags=["generation", "dedicated", "python-defined"],
    )
    deployments.append(("Generation Only Flow", deployment_generation_only))
    
    # Deployment para pipeline completo con múltiples modelos
    deployment_multi_model_pipeline = entry_point_flow.serve(
        name="py-multi-model-training",
        description="Pipeline para entrenar múltiples tipos de modelos",
        parameters={
            "flow_type": "training",
            "model_config_name": "hmm_experiment_01",  # Modelo HMM avanzado
            "model_name": "advanced_hmm_model"
        },
        tags=["training", "hmm", "advanced", "multi-model", "python-defined"],
    )
    deployments.append(("Multi-Model Pipeline", deployment_multi_model_pipeline))
    
    return deployments


def main():
    """
    Función principal para crear y mostrar información sobre deployments.
    """
    print("🚀 Configurando deployments definidos en Python para Prefect 3...")
    
    try:
        deployments = create_deployments()
        
        print(f"\n📊 Resumen:")
        print(f"✅ Deployments creados: {len(deployments)}")
        
        print("\n🎯 Deployments configurados desde deploy.py")
        print("💡 En Prefect 3, los deployments se crean usando flow.serve() y se ejecutan automáticamente")
        print("📝 Puedes ejecutar estos deployments desde la UI de Prefect")
        
        print("\n📋 Deployments disponibles:")
        for description, deployment in deployments:
            print(f"   - {description}")
            
        print("\n🔄 Para usar estos deployments:")
        print("1. Ejecuta este script: python deploy.py")
        print("2. Los flows estarán disponibles en la UI de Prefect")
        print("3. También puedes ejecutarlos directamente desde Python")
        
        # Ejemplo de ejecución directa
        print("\n💡 Ejemplo de ejecución directa:")
        print("   from src.flows.main import entry_point_flow")
        print("   entry_point_flow('training', model_config_name='gmm_default')")
        
    except Exception as e:
        print(f"❌ Error al configurar deployments: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Deployments configurados exitosamente!")
    else:
        print("\n❌ Falló la configuración de deployments.") 