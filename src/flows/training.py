import time
from pathlib import Path
from prefect import flow, task

@task
def train_dummy_model(data_path: str, model_name: str):
    """
    Dummy task wich simulates the model training.
    """
    print(f"🏋️ Entrenando el modelo '{model_name}' con datos de '{data_path}'...")
    for i in range(5):
        print(f"Entrenando... {i+1}/5")
        time.sleep(2)

    print(f"🏋️ Entrenamiento del modelo '{model_name}' completado.")

    # Create a directory for the model registry
    model_registry_path = Path(f"./model_registry/{model_name}")
    model_registry_path.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model to the model registry
    model_artifact_path = model_registry_path / "dummy_model.pkl"
    with open(model_artifact_path, "w") as f:
        f.write("This is a dummy model artifact.")

    print(f"📦 Modelo '{model_name}' guardado en '{model_artifact_path}'.")
    return str(model_artifact_path)

@flow(log_prints=True)
def training_flow(data_path: str, model_name: str):
    """
    Flow to train a model.
    """
    print(f"🚀 Iniciando el flujo de entrenamiento para '{model_name}' con datos de '{data_path}'...")
    
    # Call the task to train the model
    model_artifact_path = train_dummy_model(data_path, model_name)
    
    print(f"✅ Flujo de entrenamiento completado. Modelo guardado en '{model_artifact_path}'.")
    return model_artifact_path