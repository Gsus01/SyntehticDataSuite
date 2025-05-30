import time
from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
import pandas as pd
from typing import Dict, Any, Optional

# Model imports - will grow as more models are added
from src.models.gmm_model import GMMWrapper
# from src.models.base import SyntheticModel # For type hinting if needed


@task
def train_model_task(data_path, model_name, config):
    """
    Task to train a specified model.
    """
    logger = get_run_logger()
    logger.info(f"💾 Loading data from '{data_path}'...")
    
    # Determine file type and load data
    try:
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path, sep=r"\s+")
            logger.debug(f"Loaded CSV file with shape: {df.shape}")
        elif data_path.endswith(".txt"):
            # Assuming space-separated for .txt, adjust if needed or make configurable
            df = pd.read_csv(data_path, sep=r"\s+")
            logger.debug(f"Loaded TXT file with shape: {df.shape}")
        else:
            error_msg = f"Unsupported file format: {data_path}. Use .csv or .txt"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.debug(f"Data columns: {list(df.columns)}")
        logger.debug(f"Data types: {df.dtypes.to_dict()}")
    except Exception as e:
        logger.error(f"❌ Failed to load data from '{data_path}': {str(e)}")
        raise
    
    logger.info(f"🏋️ Training model '{model_name}' of type '{config['model_type']}'...")

    # Define where the model artifact will be saved
    model_artifact_path = Path(f"./model_registry/{model_name}/model.pkl") # Standardize to .pkl
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Model will be saved to: {model_artifact_path}")

    # Instantiate and train the model based on model_type
    model_instance: Any = None # Using Any for now, could use SyntheticModel base type

    try:
        if config["model_type"] == "gmm":
            logger.debug("Initializing GMM model wrapper")
            model_instance = GMMWrapper(
                model_name=model_name, 
                model_path=str(model_artifact_path), 
                config=config
            )
            model_instance.train(df)
        elif config["model_type"] == "dummy_time_series_model":
            # Keep dummy model logic for now, or adapt it to the new structure
            logger.info(f"Simulating training for dummy model '{model_name}'")
            for i in range(5):
                logger.debug(f"Training progress... {i+1}/5")
                time.sleep(1) # Reduced sleep time
            # Create a dummy artifact for the dummy model
            with open(model_artifact_path, "w") as f:
                f.write("This is a dummy model artifact.")
            logger.info(f"📦 Dummy model '{model_name}' saved to '{model_artifact_path}'")
        else:
            error_msg = f"Unknown model type: {config['model_type']}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Save the model (if not dummy or if save is handled by the model class)
        if hasattr(model_instance, 'save_model'):
            saved_path = model_instance.save_model()
            logger.info(f"📦 Model '{model_name}' saved to '{saved_path}'")
            return saved_path
        elif config["model_type"] == "dummy_time_series_model":
            return str(model_artifact_path) # Dummy model already "saved"
        else:
            # Should not happen if model_type is known
            error_msg = "Model was not saved"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"❌ Failed to train model '{model_name}': {str(e)}")
        raise


@flow(log_prints=True)
def training_flow(data_path, model_name, config):
    """
    Flow to train a model based on the provided configuration.
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting training flow for '{model_name}' with config: {config['model_type']}")
    logger.debug(f"Data path: {data_path}")
    logger.debug(f"Configuration: {config}")
    
    try:
        model_artifact_path = train_model_task(data_path, model_name, config)
        logger.info(f"✅ Training flow completed successfully. Model saved to '{model_artifact_path}'")
        return model_artifact_path
    except Exception as e:
        logger.error(f"❌ Training flow failed: {str(e)}")
        raise