import time
import pandas as pd
from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
from typing import Dict, Any, Optional

# Model imports
from src.models.gmm_model import GMMWrapper
from src.models.hmm_model import HMMWrapper
from src.core.config_manager import get_config, construct_path, get_model_config
from src.core.model_selector import select_model
# from src.models.base import SyntheticModel


@task
def generate_data_task(model_path: str, model_name: str, config: Dict[str, Any], n_samples: int, output_path: Optional[str] = None):
    """
    Task to generate synthetic data using a trained model.
    """
    logger = get_run_logger()
    logger.info(f"🧠 Loading model '{model_name}' from '{model_path}' to generate {n_samples} samples")
    
    model_instance: Any = None
    
    # Use dynamic output path if not provided
    if output_path is None:
        output_filename = f"synthetic_output_{model_name}.csv"
        output_path = str(construct_path("synthetic_output_dir", output_filename))
    
    logger.debug(f"Output will be saved to: {output_path}")
    logger.debug(f"Model configuration: {config}")

    try:
        if config["model_type"] == "gmm":
            logger.debug("Initializing GMM model wrapper")
            model_instance = GMMWrapper(
                model_name=model_name, 
                model_path=model_path, 
                config=config
            )
            model_instance.load_model()
            df_synthetic = model_instance.sample(n_samples)
        elif config["model_type"] == "hmm":
            logger.debug("Initializing HMM model wrapper")
            model_instance = HMMWrapper(
                model_name=model_name,
                model_path=model_path,
                config=config
            )
            model_instance.load_model()
            df_synthetic = model_instance.sample(n_samples)
        elif config["model_type"] == "dummy_time_series_model":
            logger.info(f"Simulating data generation for dummy model '{model_name}'")
            logger.debug("Creating dummy time series data")
            time.sleep(2) # Reduced sleep time
            data = {
                'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=n_samples, freq='h')),
                'value': pd.Series(range(n_samples)) + pd.Series(range(n_samples)).apply(lambda x: x * 0.1),
                'model_name': model_name
            }
            df_synthetic = pd.DataFrame(data)
            logger.debug(f"Created dummy data with shape: {df_synthetic.shape}")
        else:
            error_msg = f"Unknown model type: {config['model_type']}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Save generated data
        logger.debug(f"Saving synthetic data to {output_path}")
        df_synthetic.to_csv(output_path, index=False)
        logger.info(f"✅ Synthetic data ({n_samples} samples) generated by '{model_name}' and saved to '{output_path}'")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to generate data with model '{model_name}': {str(e)}")
        raise


@flow(log_prints=True)
def generation_flow(model_name: Optional[str] = None, model_config_name: Optional[str] = None, n_samples: Optional[int] = None, output_path: Optional[str] = None):
    """
    Flow to generate synthetic data using configuration from config.yaml.
    
    Args:
        model_name: Name of the trained model to load. If None, uses model_config_name.
        model_config_name: Name of model configuration in config.yaml. If None, uses default.
        n_samples: Number of samples to generate. If None, uses default from config.
        output_path: Where to save the output. If None, uses dynamic path.
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting generation flow with config-based approach")
    
    try:
        # Get main configuration
        main_config = get_config()
        
        # Determine model name for loading the trained model
        if model_name is None:
            if model_config_name is None:
                model_config_name = main_config.get("flows", {}).get("generation", {}).get("default_model_to_use", "gmm_default")
                logger.info(f"🤖 Using default model config from YAML: '{model_config_name}'")
            model_name = model_config_name
            logger.info(f"🤖 Using model name from config: '{model_name}'")
        else:
            logger.info(f"👨‍💻 Using specified model name: '{model_name}'")
        
        # If model_config_name is still None but model_name is specified, 
        # try to infer the model type from the model file
        if model_config_name is None and model_name is not None:
            logger.info(f"🔍 Model config not specified, attempting to infer from model name '{model_name}'")
            
            # Check if model file exists and try to determine type
            model_path = str(construct_path("models_dir", f"{model_name}.pkl"))
            if Path(model_path).exists():
                # Try to infer model type from model name patterns
                if 'hmm' in model_name.lower():
                    model_config_name = "hmm_default"
                    logger.info(f"🧠 Inferred HMM model type, using config: '{model_config_name}'")
                elif 'gmm' in model_name.lower():
                    model_config_name = "gmm_default"
                    logger.info(f"🧠 Inferred GMM model type, using config: '{model_config_name}'")
                else:
                    # Default fallback - try to inspect the model file
                    try:
                        import pickle
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                            if len(model_data) == 3:  # HMM format: (model, columns, metadata)
                                model_config_name = "hmm_default"
                                logger.info(f"🔍 Detected HMM model format (3 elements), using config: '{model_config_name}'")
                            elif len(model_data) == 2:  # GMM format: (model, columns)
                                model_config_name = "gmm_default"
                                logger.info(f"🔍 Detected GMM model format (2 elements), using config: '{model_config_name}'")
                            else:
                                logger.warning(f"⚠️ Unknown model format, defaulting to GMM config")
                                model_config_name = "gmm_default"
                    except Exception as e:
                        logger.warning(f"⚠️ Could not inspect model file, defaulting to GMM: {str(e)}")
                        model_config_name = "gmm_default"
            else:
                logger.warning(f"⚠️ Model file not found, defaulting to GMM config")
                model_config_name = "gmm_default"
        
        # If still no model_config_name, use default
        if model_config_name is None:
            model_config_name = main_config.get("flows", {}).get("generation", {}).get("default_model_to_use", "gmm_default")
            logger.info(f"🤖 Using final default model config: '{model_config_name}'")
        
        # Get model configuration
        config = select_model(model_config_name)
        
        # Determine number of samples
        if n_samples is None:
            n_samples = main_config.get("flows", {}).get("generation", {}).get("n_samples_to_generate", 1000)
            logger.info(f"🤖 Using default n_samples from config: {n_samples}")
        else:
            logger.info(f"👨‍💻 Using specified n_samples: {n_samples}")
        
        # At this point, both model_name and n_samples should be strings/ints
        assert model_name is not None, "model_name should not be None at this point"
        assert n_samples is not None, "n_samples should not be None at this point"
        
        # Construct model path
        model_path = str(construct_path("models_dir", f"{model_name}.pkl"))
        logger.debug(f"Looking for model at: {model_path}")
        
        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at '{model_path}'. Please train the model first.")
        
        logger.debug(f"Selected configuration: {config}")
        logger.debug(f"Generation parameters: model={model_name}, n_samples={n_samples}")
        
        # Generate data
        output_path = generate_data_task(model_path, model_name, config, n_samples, output_path)
        
        logger.info(f"✅ Generation flow completed successfully. Data saved to '{output_path}'")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Generation flow failed: {str(e)}")
        raise


# Legacy flow for backward compatibility
@flow(log_prints=True)
def generation_flow_legacy(model_path: str, model_name: str, config: Dict[str, Any], n_samples: int):
    """
    Legacy generation flow for backward compatibility.
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting legacy generation flow with '{model_name}' (type: {config['model_type']}) for {n_samples} samples")
    logger.debug(f"Model path: {model_path}")
    
    try:
        output_path = generate_data_task(model_path, model_name, config, n_samples)
        logger.info(f"✅ Generation flow completed successfully. Data saved to '{output_path}'")
        return output_path
    except Exception as e:
        logger.error(f"❌ Generation flow failed: {str(e)}")
        raise
