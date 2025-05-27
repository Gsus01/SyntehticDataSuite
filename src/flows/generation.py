import time
import pandas as pd
from pathlib import Path
from prefect import flow, task
from typing import Dict, Any

# Model imports
from src.models.gmm_model import GMMWrapper
# from src.models.base import SyntheticModel


@task
def generate_data_task(model_path: str, model_name: str, config: Dict[str, Any], n_samples: int):
    """
    Task to generate synthetic data using a trained model.
    """
    print(f"🧠 Cargando modelo '{model_name}' desde '{model_path}' para generar {n_samples} muestras.")
    
    model_instance: Any = None
    output_filename = f"synthetic_output_{model_name}.csv" # Dynamic output filename
    output_path = Path("./data") / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config["model_type"] == "gmm":
        # For GMM, config isn't strictly needed for loading/sampling if all info is in the .pkl
        # but good to have for consistency and if some sampling params come from config
        model_instance = GMMWrapper(
            model_name=model_name, 
            model_path=model_path, 
            config=config # Pass config, might be used for sampling parameters in future
        )
        model_instance.load_model()
        df_synthetic = model_instance.sample(n_samples)
    elif config["model_type"] == "dummy_time_series_model":
        print(f"Simulating data generation for dummy model '{model_name}'")
        time.sleep(2) # Reduced sleep time
        data = {
            'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=n_samples, freq='h')),
            'value': pd.Series(range(n_samples)) + pd.Series(range(n_samples)).apply(lambda x: x * 0.1),
            'model_name': model_name
        }
        df_synthetic = pd.DataFrame(data)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {config['model_type']}")

    # Guardar los datos generados
    df_synthetic.to_csv(output_path, index=False)
    print(f"✅ Datos sintéticos ({n_samples} muestras) generados por '{model_name}' y guardados en '{output_path}'.")
    return str(output_path)


@flow(log_prints=True)
def generation_flow(model_path: str, model_name: str, config: Dict[str, Any], n_samples: int):
    """
    Flow to generate synthetic data using a specific model.
    """
    print(f"🚀 Iniciando el flujo de generación con '{model_name}' (tipo: {config['model_type']}) para {n_samples} muestras...")
    
    output_path = generate_data_task(model_path, model_name, config, n_samples)
    
    print(f"✅ Flujo de generación completado. Datos guardados en '{output_path}'.")
    return output_path
