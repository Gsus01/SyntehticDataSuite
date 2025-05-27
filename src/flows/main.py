import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from prefect import flow, task
from prefect.deployments import run_deployment
import time
from prefect.client import get_client
import asyncio
from prefect.flow_runs import wait_for_flow_run

from src.core.model_selector import select_model

# Assuming your flows are deployed with names that match their function names
# e.g., training_flow is deployed as 'training-flow/training-deployment'

@task
def preprocess_data(raw_data_path: str) -> str:
    """
    Dummy task that simulates the preprocessing of data.
    """
    print(f"🔧 Preprocesando datos de '{raw_data_path}'...")
    time.sleep(2) # Reduced sleep
    
    processed_path = Path("./data/processed_input.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    # Copiar en vez de mover para no borrar el raw
    try:
        with open(raw_data_path, "rb") as src, open(processed_path, "wb") as dst:
            dst.write(src.read())
        print(f"✅ Datos preprocesados y guardados en '{processed_path}'.")
    except FileNotFoundError:
        print(f"❌ Error: Archivo raw no encontrado en '{raw_data_path}'. Usando datos de ejemplo.")
        # Create a dummy processed file if raw_data_path is not found for flow to proceed
        # This should be configured by the user with actual data for GMM to work.
        dummy_df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6], 'col3': [7,8,9]})
        dummy_df.to_csv(processed_path, index=False)
        print(f"📝 Creado archivo de datos procesados dummy en '{processed_path}'. Por favor, reemplace con datos reales.")
    return str(processed_path)

@task
def check_if_model_exists(model_name: str, model_registry_base_path: str = "./model_registry") -> str | None:
    """Verifies if the model already exists in the model registry (checks for model.pkl)."""
    model_path = Path(model_registry_base_path) / model_name / "model.pkl"
    if model_path.exists():
        print(f"💡 Modelo '{model_name}' ya existe en '{model_path}'. No se necesita re-entrenamiento.")
        return str(model_path)
    print(f"🤔 Modelo '{model_name}' no encontrado en '{model_path}'. Se necesita entrenamiento.")
    return None


@flow(log_prints=True)
def main_orchestration_flow(raw_data_path: str, config: dict):
    """
    Main orchestration flow for data processing, model training, and synthetic data generation.
    """
    print("🏁 Iniciando el pipeline de generación de datos sintéticos.")

    # 1. Preprocesar datos
    processed_data_path = preprocess_data(raw_data_path)

    # 2. Seleccionar el tipo de modelo y su configuración específica
    # The main `config` should contain general settings, and model-specific ones like `gmm_columns_to_use`
    model_params_selected = select_model(config) # MODIFIED: model_config_selected -> model_params_selected
    base_model_name = config.get("model_name_prefix", model_params_selected['model_type']) # MODIFIED

    # Handle multiple model training runs if desired (e.g., for A/B testing or different params)
    # For simplicity, let's assume for now we train one primary model unless specified otherwise
    # The logic for training_run_1, training_run_2 needs to be clearer on intent.
    # Are they different models, or same model type with different params, or just two instances?

    # Let's define a list of models to train. Each entry could be a (name_suffix, specific_config_override)
    # For now, we'll train one model identified by base_model_name and use model_params_selected for it. # MODIFIED
    # If you want multiple GMMs, you'd iterate here, potentially adjusting model_params_selected for each. # MODIFIED
    
    models_to_train_and_generate = [] # List to store (model_name, model_path, model_run_config) for generation

    # Example: Training a single model (or the first of multiple)
    # You could expand this to loop through a list of model configurations from `config`
    model_names_to_train = [base_model_name] # Can be extended e.g. [f"{base_model_name}_v1", f"{base_model_name}_v2"]
    
    trained_model_paths = {} # To store paths of trained models

    for i, model_name_instance in enumerate(model_names_to_train):
        print(f"\n--- Procesando Modelo #{i+1}: {model_name_instance} ---")
        # Allow config to specify different parameters for each model instance if needed.
        # For now, using the same selected config for all instances.
        current_model_params = model_params_selected.copy() # MODIFIED: current_model_config -> current_model_params
        # Example: if you had specific overrides in your main config for multiple models:
        # if f"model_instance_{i+1}_params" in config:
        # current_model_params.update(config[f"model_instance_{i+1}_params"])

        # 3. Verificar si el modelo ya existe
        # The model_name_instance already includes any suffix like _1, _2 if you decide to use them.
        existing_model_path = check_if_model_exists(model_name_instance)
        
        final_model_path_for_instance = existing_model_path

        if existing_model_path is None:
            print(f"🚦 Disparando flujo de entrenamiento para '{model_name_instance}'...")
            training_run = run_deployment(
                name="training-flow/training-deployment", # Ensure this matches your deployment name
                parameters={
                    "data_path": processed_data_path,
                    "model_name": model_name_instance,
                    "config": current_model_params # Fixed: changed from model_run_config to config
                },
                timeout=0, # Non-blocking, we will wait later
            )
            # We need to wait for this run and get its result (the model path)
            async def _wait_for_training():
                async with get_client() as client:
                    print(f"⏳ Esperando a que el entrenamiento de '{model_name_instance}' (ID: {training_run.id}) termine...")
                    completed_run = await wait_for_flow_run(training_run.id, timeout=10800, poll_interval=10)
                    if completed_run.state_name == "Completed":
                        # The training_flow returns the model_artifact_path
                        # Accessing result of a subflow run is not direct via run_deployment object.
                        # The training flow itself saves the model. We construct the path here based on convention.
                        # This should ideally be returned by the flow run state if Prefect supports easy retrieval.
                        return str(Path(f"./model_registry/{model_name_instance}/model.pkl"))
                    else:
                        print(f"❌ Entrenamiento de '{model_name_instance}' falló o fue cancelado. Estado: {completed_run.state_name}")
                        return None

            final_model_path_for_instance = asyncio.run(_wait_for_training())
            if final_model_path_for_instance:
                print(f"✅ Entrenamiento para '{model_name_instance}' completado. Modelo en '{final_model_path_for_instance}'.")
            else:
                print(f"❌ No se pudo obtener la ruta del modelo para '{model_name_instance}'. Saltando generación.")
                continue # Skip generation for this model if training failed
        
        if final_model_path_for_instance:
            trained_model_paths[model_name_instance] = final_model_path_for_instance
            models_to_train_and_generate.append((model_name_instance, final_model_path_for_instance, current_model_params)) # MODIFIED

    # 5. Generar datos sintéticos para cada modelo entrenado/existente
    n_samples = config.get("n_samples", 100)
    generated_data_paths = {}

    if not models_to_train_and_generate:
        print("⚠️ No hay modelos entrenados o existentes para la generación de datos.")
    else:
        print(f"\n--- Iniciando Generación de Datos Sintéticos para {len(models_to_train_and_generate)} modelo(s) ---")

    generation_runs_info = [] # To store info for waiting

    for model_name_gen, model_path_gen, model_params_gen in models_to_train_and_generate: # MODIFIED: model_config_gen -> model_params_gen
        print(f"🧬 Disparando flujo de generación para '{model_name_gen}' usando modelo de '{model_path_gen}'...")
        generation_run = run_deployment(
            name="generation-flow/generation-deployment", # Ensure this matches your deployment name
            parameters={
                "model_path": model_path_gen,
                "model_name": model_name_gen, # Pass model_name for context and output naming
                "config": model_params_gen, # Fixed: changed from model_run_config to config
                "n_samples": n_samples
            },
            timeout=0, # Non-blocking for now, will wait if needed
        )
        generation_runs_info.append((model_name_gen, generation_run.id))

    # Wait for all generation flows to complete if you want to ensure they are done before main flow finishes
    async def _wait_for_generation():
        async with get_client() as client:
            for model_name_gen, run_id in generation_runs_info:
                print(f"⏳ Esperando a que la generación de '{model_name_gen}' (ID: {run_id}) termine...")
                completed_run = await wait_for_flow_run(run_id, timeout=3600, poll_interval=10)
                if completed_run.state_name == "Completed":
                    # Similar to training, constructing path based on convention
                    output_file = f"synthetic_output_{model_name_gen}.csv"
                    gen_path = str(Path("./data") / output_file)
                    generated_data_paths[model_name_gen] = gen_path
                    print(f"✅ Generación para '{model_name_gen}' completada. Datos en '{gen_path}'.")
                else:
                    print(f"❌ Generación para '{model_name_gen}' falló o fue cancelada. Estado: {completed_run.state_name}")

    if generation_runs_info:
        asyncio.run(_wait_for_generation())

    print("\n🏁 Pipeline de generación de datos sintéticos finalizado.")
    if generated_data_paths:
        print("Resumen de datos generados:")
        for name, path in generated_data_paths.items():
            print(f"  - Modelo '{name}': {path}")
    else:
        print("No se generaron datos sintéticos en esta ejecución.")