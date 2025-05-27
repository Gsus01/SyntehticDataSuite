from prefect import task
import time

@task
def select_model(config: dict) -> dict:
    """
    Decide which model to use and return its configuration.
    """

    model_type = config.get("model_type", "auto")
    
    if model_type == "auto":
        print("🤖 Decidiendo automáticamente...")
        time.sleep(5)
        print("🤖 Seleccionado el modelo de ejemplo: 'dummy_time_series_model'.")
        return {
            "model_type": "dummy_time_series_model", 
        }
    elif model_type == "gmm":
        print(f"👨‍💻 Selección manual: Usaremos el modelo GMM.")
        
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
            print(f"⚠️ Usando columnas por defecto para GMM: {default_columns}. Especifica 'gmm_columns_to_use' para personalizar.")
        else:
            print(f"✅ Usando columnas especificadas para GMM: {columns_to_use}")
            
        return gmm_config
    
    else:
        print(f"👨‍💻 Selección manual: Usaremos el modelo '{model_type}'.")
        return {"model_type": model_type, **config}