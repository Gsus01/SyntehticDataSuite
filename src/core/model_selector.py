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
    
    else:
        logger.info(f"👨‍💻 Manual selection: Using model '{model_type}'")
        final_config = {"model_type": model_type, **config}
        logger.debug(f"Final configuration for model '{model_type}': {final_config}")
        return final_config