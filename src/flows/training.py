import time
from pathlib import Path
from prefect import flow, task
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
    print(f"💾 Cargando datos desde '{data_path}'...")
    # Determine file type and load data
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".txt"):
        # Assuming space-separated for .txt, adjust if needed or make configurable
        df = pd.read_csv(data_path, delim_whitespace=True) 
    else:
        raise ValueError(f"Formato de archivo no soportado: {data_path}. Usar .csv o .txt")
    
    print(f"🏋️ Entrenando el modelo '{model_name}' de tipo '{config['model_type']}'...")

    # Define where the model artifact will be saved
    model_artifact_path = Path(f"./model_registry/{model_name}/model.pkl") # Standardize to .pkl
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate and train the model based on model_type
    model_instance: Any = None # Using Any for now, could use SyntheticModel base type

    if config["model_type"] == "gmm":
        model_instance = GMMWrapper(
            model_name=model_name, 
            model_path=str(model_artifact_path), 
            config=config
        )
        model_instance.train(df)
    elif config["model_type"] == "dummy_time_series_model":
        # Keep dummy model logic for now, or adapt it to the new structure
        print(f"Simulating training for dummy model '{model_name}'")
        for i in range(5):
            print(f"Entrenando... {i+1}/5")
            time.sleep(1) # Reduced sleep time
        # Create a dummy artifact for the dummy model
        with open(model_artifact_path, "w") as f:
            f.write("This is a dummy model artifact.")
        print(f"📦 Modelo Dummy '{model_name}' guardado en '{model_artifact_path}'.")
    else:
        raise ValueError(f"Tipo de modelo desconocido: {config['model_type']}")

    # Save the model (if not dummy or if save is handled by the model class)
    if hasattr(model_instance, 'save_model'):
        saved_path = model_instance.save_model()
        print(f"📦 Modelo '{model_name}' guardado en '{saved_path}'.")
        return saved_path
    elif config["model_type"] == "dummy_time_series_model":
        return str(model_artifact_path) # Dummy model already "saved"
    else:
        # Should not happen if model_type is known
        raise Exception("Model was not saved.") 


@flow
def training_flow(data_path, model_name, config):
    """
    Flow to train a model based on the provided configuration.
    """
    print(f"🚀 Iniciando el flujo de entrenamiento para '{model_name}' con config: {config['model_type']}...")
    
    model_artifact_path = train_model_task(data_path, model_name, config)
    
    print(f"✅ Flujo de entrenamiento completado. Modelo guardado en '{model_artifact_path}'.")
    return model_artifact_path