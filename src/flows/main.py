
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from prefect import flow, task
from prefect.deployments import run_deployment
import time
from prefect.client import get_client
import asyncio
from prefect.flow_runs import wait_for_flow_run

from src.core.model_selector import select_model


@task
def preprocess_data(raw_data_path: str) -> str:
    """
    Dummy task that simulates the preprocessing of data.
    """
    print(f"🔧 Preprocesando datos de '{raw_data_path}'...")
    # Simulate some preprocessing
    time.sleep(5)
    
    processed_path = "./data/processed_input.csv"
    # Copiar en vez de mover para no borrar el raw
    with open(raw_data_path, "rb") as src, open(processed_path, "wb") as dst:
        dst.write(src.read())
    print(f"✅ Datos preprocesados y guardados en '{processed_path}'.")
    return processed_path

@task
def check_if_model_exists(model_name: str) -> str | None:
    """Verifies if the model already exists in the model registry."""
    model_path = Path(f"./model_registry/{model_name}/model.pkl")
    if model_path.exists():
        print(f"💡 Modelo '{model_name}' ya existe en '{model_path}'. No se necesita re-entrenamiento.")
        return str(model_path)
    print(f"🤔 Modelo '{model_name}' no encontrado. Se necesita entrenamiento.")
    return None


@flow(log_prints=True)
def main_orchestration_flow(raw_data_path: str, config: dict):
    """
    Main orchestration flow for data processing and model training.
    """
    print("🏁 Iniciando el pipeline de generación de datos sintéticos.")

    # 1. Preprocesar datos
    processed_data_path = preprocess_data(raw_data_path)

    # 2. Seleccionar el modelo
    model_name = select_model(config)

    # 3. Verificar si el modelo ya existe
    model_path = check_if_model_exists(model_name)
    
    # 4. Si el modelo no existe, entrenarlo y esperar a que termine
    if model_path is None:
        print("Triggering training_flow deployment...")
        training_run_1 = run_deployment(
            name="training-flow/training-deployment",
            parameters={
                "data_path": processed_data_path,
                "model_name": f"{model_name}_1"
                },
            timeout=0,
        )

        training_run_2 = run_deployment(
            name="training-flow/training-deployment",
            parameters={
                "data_path": processed_data_path,
                "model_name": f"{model_name}_2"
                },
            timeout=0,
        )
        
        
        async def _wait():
            async with get_client() as client:
                await wait_for_flow_run(training_run_1.id, timeout=10800, poll_interval=5)
                await wait_for_flow_run(training_run_2.id, timeout=10800, poll_interval=5)

        model_path_1 = str(Path(f"./model_registry/{model_name}_1/model.pkl"))
        model_path_2 = str(Path(f"./model_registry/{model_name}_2/model.pkl"))

        asyncio.run(_wait())
        print(f"✅ Entrenamiento completado. Modelos guardados en '{model_path_1}' y '{model_path_2}'.")
        
        
    # 5. Generar datos sintéticos SOLO después de entrenamiento
    n_samples = config.get("n_samples", 100)
    generation_run = run_deployment(
        name="generation-flow/generation-deployment",
        parameters={"model_path": model_path_1, "n_samples": n_samples},
        timeout=None,  # Espera a que termine
    )

    # El objeto devuelto por run_deployment es FlowRun, no tiene atributo 'result'.
    # Para obtener el resultado, hay que leer el output del flow hijo, pero Prefect 2/3 no lo hace directo.
    # Como workaround, mostramos la ruta esperada del output.
    output_path = "./data/synthetic_output.csv"
    print(f"✅ Flujo de generacion completado. Datos guardados en '{output_path}'.")