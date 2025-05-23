from prefect import task
import time

@task
def select_model(config: dict) -> str:
    """
    Decide which model to use.
    """

    model_name = config.get("model", "auto")
    
    if model_name == "auto":
        print("🤖 Decidiendo automáticamente...")
        time.sleep(5)
        print("🤖 Seleccionado el modelo 'dummy_time_series_model'.")
        return "dummy_time_series_model"
    
    print(f"👨‍💻 Selección manual: Usaremos el modelo '{model_name}'.")
    return model_name