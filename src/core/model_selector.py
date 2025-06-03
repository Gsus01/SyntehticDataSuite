from prefect import task
from prefect.logging import get_run_logger
import time
from typing import Optional
from src.core.config_manager import get_config, get_model_config

@task
def select_model(model_config_name: Optional[str] = None) -> dict:
    """
    Decide which model to use and return its configuration from config.yaml.
    
    Args:
        model_config_name: Name of the model configuration in config.yaml.
                          If None, uses default from flows.training.default_model_to_train
    """
    logger = get_run_logger()
    
    # Get main configuration
    config = get_config()
    
    # Determine which model to use
    if model_config_name is None:
        model_config_name = config.get("flows", {}).get("training", {}).get("default_model_to_train", "gmm_default")
        logger.info(f"🤖 Using default model from config: '{model_config_name}'")
    else:
        logger.info(f"👨‍💻 Manual selection: Using model '{model_config_name}'")
    
    # At this point, model_config_name is guaranteed to be a string
    assert model_config_name is not None, "model_config_name should not be None at this point"
    
    logger.debug(f"Model selection input: {model_config_name}")
    
    try:
        # Get model configuration from YAML
        model_config = get_model_config(model_config_name)
        model_type = model_config.get("type", "").lower()
        model_params = model_config.get("params", {})
        
        logger.info(f"📋 Selected model type: {model_type}")
        logger.debug(f"Model parameters from config: {model_params}")
        
        # Create the final configuration based on model type
        if model_type == "gmm":
            logger.info(f"✅ Configuring GMM model from YAML")
            
            final_config = {
                "model_type": "gmm",
                "gmm_n_components": model_params.get("n_components", 3),
                "gmm_covariance_type": model_params.get("covariance_type", "diag"),
                "gmm_max_iter": model_params.get("max_iter", 100),
                "gmm_random_state": model_params.get("random_state", 42)
            }
            
            # Add optional parameters if present
            if "tol" in model_params:
                final_config["gmm_tol"] = model_params["tol"]
            if "init_params" in model_params:
                final_config["gmm_init_params"] = model_params["init_params"]
            if "columns_to_use" in model_params:
                final_config["gmm_columns_to_use"] = model_params["columns_to_use"]
                logger.info(f"✅ Using specified columns for GMM: {model_params['columns_to_use']}")
            else:
                logger.info("📋 GMM will use all numeric columns (auto-selection)")
            
        elif model_type == "hmm":
            logger.info(f"✅ Configuring HMM model from YAML")
            
            final_config = {
                "model_type": "hmm",
                "hmm_n_states": model_params.get("n_components", 3),  # Note: n_components maps to n_states for HMM
                "hmm_covariance_type": model_params.get("covariance_type", "diag"),
                "hmm_n_iter": model_params.get("n_iter", 100),
                "hmm_random_state": model_params.get("random_state", 42)
            }
            
            # Add optional HMM-specific parameters
            if "tol" in model_params:
                final_config["hmm_tol"] = model_params["tol"]
            if "algorithm" in model_params:
                final_config["hmm_algorithm"] = model_params["algorithm"]
            if "columns_to_use" in model_params:
                final_config["hmm_columns_to_use"] = model_params["columns_to_use"]
                logger.info(f"✅ Using specified columns for HMM: {model_params['columns_to_use']}")
            else:
                logger.info("📋 HMM will auto-select all numeric columns")
            if "sequence_column" in model_params:
                final_config["hmm_sequence_column"] = model_params["sequence_column"]
                logger.info(f"🕒 Using sequence column for HMM: {model_params['sequence_column']}")
                
        else:
            # For future model types or custom configurations
            logger.info(f"✅ Configuring {model_type} model from YAML")
            final_config = {
                "model_type": model_type,
                **model_params  # Include all parameters as-is
            }
        
        logger.debug(f"Final model configuration: {final_config}")
        return final_config
        
    except ValueError as e:
        logger.error(f"❌ Model configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in model selection: {str(e)}")
        raise