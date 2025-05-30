from prefect import task
from prefect.logging import get_run_logger
import time

@task
def select_model(config: dict) -> dict:
    """
    Decide which model to use and return its configuration.
    """
    logger = get_run_logger()
    model_type = config.get("model_type", "auto")
    
    logger.debug(f"Model selection input config: {config}")
    logger.info(f"Model selection mode: {model_type}")
    
    if model_type == "auto":
        logger.info("🤖 Deciding automatically...")
        time.sleep(5)
        selected_model = "dummy_time_series_model"
        logger.info(f"🤖 Selected example model: '{selected_model}'")
        logger.debug("Auto-selection completed with dummy time series model")
        return {
            "model_type": selected_model, 
        }
    elif model_type == "gmm":
        logger.info(f"👨‍💻 Manual selection: Using GMM model")
        
        # Default columns to use if not specified (matches dummy data structure)
        default_columns = ["col1", "col2", "col3"]
        columns_to_use = config.get("gmm_columns_to_use", default_columns)
        
        gmm_config = {
            "model_type": "gmm",
            "gmm_columns_to_use": columns_to_use,
            "gmm_n_components": config.get("gmm_n_components", 3),
            "gmm_covariance_type": config.get("gmm_covariance_type", "diag"),
            "gmm_max_iter": config.get("gmm_max_iter", 100),
            "gmm_random_state": config.get("gmm_random_state", 42)
        }
        
        if columns_to_use == default_columns:
            logger.warning(f"⚠️ Using default columns for GMM: {default_columns}. Specify 'gmm_columns_to_use' to customize")
        else:
            logger.info(f"✅ Using specified columns for GMM: {columns_to_use}")
        
        logger.debug(f"Final GMM configuration: {gmm_config}")
        return gmm_config
    
    elif model_type == "hmm":
        logger.info(f"👨‍💻 Manual selection: Using HMM model")
        
        # HMM configuration with auto-selection support
        hmm_config = {
            "model_type": "hmm",
            "hmm_n_states": config.get("hmm_n_states", 3),
            "hmm_covariance_type": config.get("hmm_covariance_type", "diag"),
            "hmm_n_iter": config.get("hmm_n_iter", 100),
            "hmm_random_state": config.get("hmm_random_state", 42),
        }
        
        # Optional: add columns if specified (otherwise auto-selection will be used)
        if "hmm_columns_to_use" in config:
            hmm_config["hmm_columns_to_use"] = config["hmm_columns_to_use"]
            logger.info(f"✅ Using specified columns for HMM: {config['hmm_columns_to_use']}")
        else:
            logger.info("📋 HMM will auto-select all numeric columns")
        
        # Optional: add sequence column if specified
        if "hmm_sequence_column" in config:
            hmm_config["hmm_sequence_column"] = config["hmm_sequence_column"]
            logger.info(f"🕒 Using sequence column for HMM: {config['hmm_sequence_column']}")
        
        # Add other optional HMM parameters if specified
        optional_params = ["hmm_tol", "hmm_algorithm"]
        for param in optional_params:
            if param in config:
                hmm_config[param] = config[param]
        
        logger.debug(f"Final HMM configuration: {hmm_config}")
        return hmm_config
    
    else:
        logger.info(f"👨‍💻 Manual selection: Using model '{model_type}'")
        final_config = {"model_type": model_type, **config}
        logger.debug(f"Final configuration for model '{model_type}': {final_config}")
        return final_config