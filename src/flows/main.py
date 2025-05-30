import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
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
    logger = get_run_logger()
    logger.info(f"🔧 Preprocessing data from '{raw_data_path}'...")
    time.sleep(2)
    
    processed_path = Path("./data/processed_input.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Processing data to: {processed_path}")
    
    try:
        with open(raw_data_path, "rb") as src, open(processed_path, "wb") as dst:
            dst.write(src.read())
        logger.info(f"✅ Data preprocessed and saved to '{processed_path}'")
        
    except FileNotFoundError:
        logger.warning(f"❌ Error: Raw file not found at '{raw_data_path}'. Using example data")

    except Exception as e:
        logger.error(f"❌ Failed to preprocess data: {str(e)}")
        raise
    
    return str(processed_path)

@task
def check_if_model_exists(model_name: str, model_registry_base_path: str = "./model_registry") -> str | None:
    """Verifies if the model already exists in the model registry (checks for model.pkl)."""
    logger = get_run_logger()
    model_path = Path(model_registry_base_path) / model_name / "model.pkl"
    
    logger.debug(f"Checking for existing model at: {model_path}")
    
    if model_path.exists():
        logger.info(f"💡 Model '{model_name}' already exists at '{model_path}'. No retraining needed")
        return str(model_path)
    
    logger.info(f"🤔 Model '{model_name}' not found at '{model_path}'. Training needed")
    return None


@flow(log_prints=True)
def main_orchestration_flow(raw_data_path: str, config: dict):
    """
    Main orchestration flow for data processing, model training, and synthetic data generation.
    """
    logger = get_run_logger()
    logger.info("🏁 Starting synthetic data generation pipeline")
    logger.debug(f"Raw data path: {raw_data_path}")
    logger.debug(f"Configuration: {config}")

    try:
        # 1. Preprocess data
        processed_data_path = preprocess_data(raw_data_path)

        # 2. Select model type and its specific configuration
        model_params_selected = select_model(config)
        base_model_name = config.get("model_name_prefix", model_params_selected['model_type'])

        logger.info(f"Selected model configuration: {model_params_selected}")
        logger.debug(f"Base model name: {base_model_name}")

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

        logger.info(f"Models to train: {model_names_to_train}")

        for i, model_name_instance in enumerate(model_names_to_train):
            logger.info(f"\n--- Processing Model #{i+1}: {model_name_instance} ---")
            # Allow config to specify different parameters for each model instance if needed.
            # For now, using the same selected config for all instances.
            current_model_params = model_params_selected.copy() # MODIFIED: current_model_config -> current_model_params
            # Example: if you had specific overrides in your main config for multiple models:
            # if f"model_instance_{i+1}_params" in config:
            # current_model_params.update(config[f"model_instance_{i+1}_params"])

            logger.debug(f"Current model parameters: {current_model_params}")

            # 3. Check if model already exists
            # The model_name_instance already includes any suffix like _1, _2 if you decide to use them.
            existing_model_path = check_if_model_exists(model_name_instance)
            
            final_model_path_for_instance = existing_model_path

            if existing_model_path is None:
                logger.info(f"🚦 Triggering training flow for '{model_name_instance}'...")
                try:
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
                            logger.info(f"⏳ Waiting for training of '{model_name_instance}' (ID: {training_run.id}) to complete...")
                            completed_run = await wait_for_flow_run(training_run.id, timeout=10800, poll_interval=10)
                            if completed_run.state_name == "Completed":
                                # The training_flow returns the model_artifact_path
                                # Accessing result of a subflow run is not direct via run_deployment object.
                                # The training flow itself saves the model. We construct the path here based on convention.
                                # This should ideally be returned by the flow run state if Prefect supports easy retrieval.
                                model_path = str(Path(f"./model_registry/{model_name_instance}/model.pkl"))
                                logger.debug(f"Training completed, model should be at: {model_path}")
                                return model_path
                            else:
                                logger.error(f"❌ Training of '{model_name_instance}' failed or was cancelled. State: {completed_run.state_name}")
                                return None

                    final_model_path_for_instance = asyncio.run(_wait_for_training())
                    if final_model_path_for_instance:
                        logger.info(f"✅ Training for '{model_name_instance}' completed. Model at '{final_model_path_for_instance}'")
                    else:
                        logger.error(f"❌ Could not get model path for '{model_name_instance}'. Skipping generation")
                        continue # Skip generation for this model if training failed
                except Exception as e:
                    logger.error(f"❌ Failed to start training for '{model_name_instance}': {str(e)}")
                    continue
            
            if final_model_path_for_instance:
                trained_model_paths[model_name_instance] = final_model_path_for_instance
                models_to_train_and_generate.append((model_name_instance, final_model_path_for_instance, current_model_params)) # MODIFIED

        # 5. Generate synthetic data for each trained/existing model
        n_samples = config.get("n_samples", 100)
        generated_data_paths = {}

        logger.info(f"Preparing to generate {n_samples} samples per model")

        if not models_to_train_and_generate:
            logger.warning("⚠️ No trained or existing models for data generation")
        else:
            logger.info(f"\n--- Starting Synthetic Data Generation for {len(models_to_train_and_generate)} model(s) ---")

        generation_runs_info = [] # To store info for waiting

        for model_name_gen, model_path_gen, model_params_gen in models_to_train_and_generate: # MODIFIED: model_config_gen -> model_params_gen
            logger.info(f"🧬 Triggering generation flow for '{model_name_gen}' using model from '{model_path_gen}'...")
            try:
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
            except Exception as e:
                logger.error(f"❌ Failed to start generation for '{model_name_gen}': {str(e)}")

        # Wait for all generation flows to complete if you want to ensure they are done before main flow finishes
        async def _wait_for_generation():
            async with get_client() as client:
                for model_name_gen, run_id in generation_runs_info:
                    logger.info(f"⏳ Waiting for generation of '{model_name_gen}' (ID: {run_id}) to complete...")
                    try:
                        completed_run = await wait_for_flow_run(run_id, timeout=3600, poll_interval=10)
                        if completed_run.state_name == "Completed":
                            # Similar to training, constructing path based on convention
                            output_file = f"synthetic_output_{model_name_gen}.csv"
                            gen_path = str(Path("./data") / output_file)
                            generated_data_paths[model_name_gen] = gen_path
                            logger.info(f"✅ Generation for '{model_name_gen}' completed. Data at '{gen_path}'")
                        else:
                            logger.error(f"❌ Generation for '{model_name_gen}' failed or was cancelled. State: {completed_run.state_name}")
                    except Exception as e:
                        logger.error(f"❌ Error waiting for generation of '{model_name_gen}': {str(e)}")

        if generation_runs_info:
            asyncio.run(_wait_for_generation())

        logger.info("\n🏁 Synthetic data generation pipeline completed")
        if generated_data_paths:
            logger.info("Summary of generated data:")
            for name, path in generated_data_paths.items():
                logger.info(f"  - Model '{name}': {path}")
        else:
            logger.warning("No synthetic data was generated in this execution")
            
        return generated_data_paths
        
    except Exception as e:
        logger.error(f"❌ Main orchestration flow failed: {str(e)}")
        raise