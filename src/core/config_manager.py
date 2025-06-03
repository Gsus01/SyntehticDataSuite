# src/core/config_manager.py
import yaml
from pathlib import Path
import os
from typing import Optional

# Asumimos que este archivo está en src/core, y config.yaml en config/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_config = None

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    global _config
    if _config is None:
        if not config_path.exists():
            # Intenta buscar en el directorio actual si no se encuentra en la ruta por defecto (útil para tests o ejecuciones locales)
            alt_config_path = Path.cwd() / "config" / "config.yaml"
            if alt_config_path.exists():
                config_path = alt_config_path
            else:
                 raise FileNotFoundError(f"El archivo de configuración no se encontró en: {config_path} ni en {alt_config_path}")
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config

def get_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    return load_config(config_path)

def get_model_config(model_config_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    cfg = get_config(config_path)
    if model_config_name not in cfg.get('models', {}):
        raise ValueError(f"Configuración del modelo '{model_config_name}' no encontrada.")
    return cfg['models'][model_config_name]

def construct_path(key: str, filename: Optional[str] = None, model_name: Optional[str] = None, base_config_name: str = "paths", config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    cfg = get_config(config_path)
    base_dir_key = cfg.get(base_config_name, {}).get(key)
    if not base_dir_key:
        raise KeyError(f"Clave de directorio '{key}' no encontrada en la sección '{base_config_name}'.")
    
    path = PROJECT_ROOT / Path(base_dir_key)
    
    if filename:
        if model_name: # Permitir formateo del nombre de archivo con el nombre del modelo
            filename = filename.format(model_name=model_name)
        path = path / filename
    
    # Crear directorio padre si no existe, sólo al construir la ruta final con nombre de archivo
    if filename:
        path.parent.mkdir(parents=True, exist_ok=True)
    else: # Si solo se pide el directorio, asegurar que ese directorio exista
        path.mkdir(parents=True, exist_ok=True)
    return path 