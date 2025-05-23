import time
import pandas as pd
from pathlib import Path
from prefect import flow, task

@task
def generate_dummy_data(model_path: str, n_samples: int):
    """
    Dummy task that simulates the generation of data.
    """
    print(f"🧠 Cargando modelo desde '{model_path}' para generar {n_samples} muestras.")
    time.sleep(5)

    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=n_samples, freq='h')),
        'value': pd.Series(range(n_samples)) + pd.Series(range(n_samples)).apply(lambda x: x * 0.1)
    }

    df = pd.DataFrame(data)

    # Guardar los datos generados
    output_path = Path("./data/synthetic_output.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Datos sintéticos generados y guardados en '{output_path}'.")
    return str(output_path)

@flow(log_prints=True)
def generation_flow(model_path: str, n_samples: int):
    """
    Flow to generate synthetic data.
    """
    print(f"🚀 Iniciando el flujo de generación de datos con el modelo '{model_path}' para {n_samples} muestras...")
    
    # Call the task to generate data
    output_path = generate_dummy_data(model_path, n_samples)
    
    print(f"✅ Flujo de generación completado. Datos guardados en '{output_path}'.")
    return output_path
