import time
from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
import pandas as pd
from typing import Dict, Any, Optional

# Model imports - will grow as more models are added
from src.models.gmm_model import GMMWrapper
from src.models.hmm_model import HMMWrapper
from src.core.config_manager import get_config, construct_path
from src.core.model_selector import select_model
# from src.models.base import SyntheticModel # For type hinting if needed


@task
def load_data_task(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Task to load training data.
    """
    logger = get_run_logger()
    
    # If no data_path provided, use default from config
    if data_path is None:
        config = get_config()
        data_filename = config.get("paths", {}).get("default_input_file", "train_FD001.csv")
        data_path = str(construct_path("base_data_dir", data_filename))
        logger.info(f"🤖 Using default data path from config: '{data_path}'")
    else:
        logger.info(f"👨‍💻 Using specified data path: '{data_path}'")
    
    logger.info(f"💾 Loading data from '{data_path}'...")
    
    # Determine file type and load data
    try:
        if data_path.endswith(".csv"):
            # Try standard CSV first (comma-separated)
            df = pd.read_csv(data_path)
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
        logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load data from '{data_path}': {str(e)}")
        raise


@task
def train_model_task(df: pd.DataFrame, model_name: str, config: dict):
    """
    Task to train a specified model.
    """
    logger = get_run_logger()
    logger.info(f"🏋️ Training model '{model_name}' of type '{config['model_type']}'...")

    # Use config_manager to construct model path
    model_artifact_path = construct_path("models_dir", "{model_name}.pkl", model_name)
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
        elif config["model_type"] == "hmm":
            logger.debug("Initializing HMM model wrapper")
            model_instance = HMMWrapper(
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
def training_flow(data_path: Optional[str] = None, model_name: Optional[str] = None, model_config_name: Optional[str] = None):
    """
    Flow to train a model using configuration from config.yaml.
    
    Args:
        data_path: Path to training data. If None, uses default from config.
        model_name: Name for the trained model. If None, uses model_config_name.
        model_config_name: Name of model configuration in config.yaml. If None, uses default.
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting training flow with config-based approach")
    
    try:
        # Get model configuration from YAML
        config = select_model(model_config_name)
        
        # Determine model name
        if model_name is None:
            model_name = model_config_name or config.get("model_type", "default_model")
            logger.info(f"🤖 Using automatic model name: '{model_name}'")
        else:
            logger.info(f"👨‍💻 Using specified model name: '{model_name}'")
        
        logger.debug(f"Selected configuration: {config}")
        
        # Load data
        df = load_data_task(data_path)
        
        # Train model
        model_artifact_path = train_model_task(df, model_name, config)
        
        logger.info(f"✅ Training flow completed successfully. Model saved to '{model_artifact_path}'")
        return model_artifact_path
        
    except Exception as e:
        logger.error(f"❌ Training flow failed: {str(e)}")
        raise


# Legacy flow for backward compatibility
@flow(log_prints=True)
def training_flow_legacy(data_path, model_name, config):
    """
    Legacy training flow for backward compatibility.
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting legacy training flow for '{model_name}' with config: {config['model_type']}")
    logger.debug(f"Data path: {data_path}")
    logger.debug(f"Configuration: {config}")
    
    try:
        df = load_data_task(data_path)
        model_artifact_path = train_model_task(df, model_name, config)
        logger.info(f"✅ Training flow completed successfully. Model saved to '{model_artifact_path}'")
        return model_artifact_path
    except Exception as e:
        logger.error(f"❌ Training flow failed: {str(e)}")
        raise