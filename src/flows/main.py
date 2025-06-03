import pandas as pd
from pathlib import Path
from prefect import flow, task
from prefect.logging import get_run_logger
from typing import Dict, Any, Optional

from src.flows.training import training_flow, training_flow_legacy
from src.flows.generation import generation_flow, generation_flow_legacy
from src.core.config_manager import get_config, construct_path


@flow(log_prints=True)
def entry_point_flow(
    flow_type: str,
    model_config_name: Optional[str] = None,
    model_name: Optional[str] = None,
    input_data_filename: Optional[str] = None,
    n_samples: Optional[int] = None,
    output_path: Optional[str] = None
):
    """
    Unified entry point flow for both training and generation using config.yaml.
    
    Args:
        flow_type: Either "training" or "generation"
        model_config_name: Name of model configuration in config.yaml
        model_name: Name for the model (for training) or name of trained model (for generation)
        input_data_filename: Input data file for training (if different from default)
        n_samples: Number of samples to generate (for generation)
        output_path: Output path for generated data (for generation)
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting entry point flow for: {flow_type}")
    logger.debug(f"Parameters: model_config_name={model_config_name}, model_name={model_name}")
    
    try:
        if flow_type == "training":
            logger.info("🏋️ Executing training flow...")
            
            # Construct data path if input_data_filename is provided
            data_path = None
            if input_data_filename:
                data_path = str(construct_path("base_data_dir", input_data_filename))
                logger.info(f"📁 Using specified data file: {data_path}")
            
            # Execute training flow
            result = training_flow(
                data_path=data_path,
                model_name=model_name,
                model_config_name=model_config_name
            )
            
            logger.info(f"✅ Training completed. Model saved to: {result}")
            return {"flow_type": "training", "model_path": result}
            
        elif flow_type == "generation":
            logger.info("🧠 Executing generation flow...")
            
            # Execute generation flow
            result = generation_flow(
                model_name=model_name,
                model_config_name=model_config_name,
                n_samples=n_samples,
                output_path=output_path
            )
            
            logger.info(f"✅ Generation completed. Data saved to: {result}")
            return {"flow_type": "generation", "output_path": result}
            
        else:
            error_msg = f"Invalid flow_type: {flow_type}. Must be 'training' or 'generation'"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"❌ Entry point flow failed: {str(e)}")
        raise


@flow(log_prints=True)
def training_only_flow(
    model_config_name: Optional[str] = None,
    model_name: Optional[str] = None,
    input_data_filename: Optional[str] = None
):
    """
    Dedicated training flow for deployment.
    """
    return entry_point_flow(
        flow_type="training",
        model_config_name=model_config_name,
        model_name=model_name,
        input_data_filename=input_data_filename
    )


@flow(log_prints=True)
def generation_only_flow(
    model_config_name: Optional[str] = None,
    model_name: Optional[str] = None,
    n_samples: Optional[int] = None,
    output_path: Optional[str] = None
):
    """
    Dedicated generation flow for deployment.
    """
    return entry_point_flow(
        flow_type="generation",
        model_config_name=model_config_name,
        model_name=model_name,
        n_samples=n_samples,
        output_path=output_path
    )


# Legacy tasks for backward compatibility
@task
def preprocess_data(raw_data_path: str) -> str:
    """
    Preprocesses data from raw format to CSV using config manager.
    """
    logger = get_run_logger()
    logger.info(f"🔄 Preprocessing data from '{raw_data_path}'...")
    
    # Use config manager for output path
    processed_data_path = str(construct_path("processed_data_dir", "processed_input.csv"))
    
    try:
        if raw_data_path.endswith(".txt"):
            # Convert from space-separated to CSV
            df = pd.read_csv(raw_data_path, sep=r"\s+")
            df.to_csv(processed_data_path, index=False)
            logger.info(f"✅ Converted TXT to CSV: {processed_data_path}")
        elif raw_data_path.endswith(".csv"):
            # Just copy/validate CSV
            df = pd.read_csv(raw_data_path)
            df.to_csv(processed_data_path, index=False)
            logger.info(f"✅ Validated and copied CSV: {processed_data_path}")
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path}")
            
        logger.debug(f"Processed data shape: {df.shape}")
        return processed_data_path
        
    except Exception as e:
        logger.error(f"❌ Failed to preprocess data: {str(e)}")
        raise


@task
def check_if_model_exists(model_name: str) -> Optional[str]:
    """
    Check if a trained model already exists using config manager.
    """
    logger = get_run_logger()
    model_path = construct_path("models_dir", f"{model_name}.pkl")
    
    if model_path.exists():
        logger.info(f"📦 Found existing model: {model_path}")
        return str(model_path)
    else:
        logger.info(f"🆕 Model not found, will need training: {model_path}")
        return None


@flow(log_prints=True)
def main_orchestration_flow(raw_data_path: str, config: dict):
    """
    Legacy main orchestration flow for backward compatibility.
    """
    logger = get_run_logger()
    logger.info("🏁 Starting legacy synthetic data generation pipeline")
    logger.debug(f"Raw data path: {raw_data_path}")
    logger.debug(f"Configuration: {config}")

    try:
        # 1. Preprocess data
        processed_data_path = preprocess_data(raw_data_path)

        # 2. Extract model configuration
        model_type = config.get("model_type", "gmm")
        base_model_name = config.get("model_name_prefix", model_type)

        logger.info(f"Model type: {model_type}, Base name: {base_model_name}")

        # 3. Check if model already exists
        existing_model_path = check_if_model_exists(base_model_name)
        
        if existing_model_path:
            final_model_path_for_instance = existing_model_path
        else:
            # Train new model using legacy training flow
            final_model_path_for_instance = training_flow_legacy(
                data_path=processed_data_path,
                model_name=base_model_name,
                config=config
            )

        # 4. Generate synthetic data using legacy generation flow
        n_samples = config.get("n_samples", 1000)
        synthetic_data_path = generation_flow_legacy(
            model_path=final_model_path_for_instance,
            model_name=base_model_name,
            config=config,
            n_samples=n_samples
        )

        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"📊 Synthetic data saved to: {synthetic_data_path}")
        
        return {
            "model_path": final_model_path_for_instance,
            "synthetic_data_path": synthetic_data_path,
            "n_samples": n_samples
        }

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise 