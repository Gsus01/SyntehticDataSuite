import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from pathlib import Path
from typing import List, Tuple, Any, Optional, Union
import logging

from src.models.base import SyntheticModel
from src.core.config_manager import construct_path

# Set up logger for non-Prefect contexts
logger = logging.getLogger(__name__)

class GMMWrapper(SyntheticModel):
    """
    Wrapper for scikit-learn's GaussianMixture model.
    """
    def __init__(self, model_name: str, model_path: Optional[str] = None, config: Optional[dict] = None):
        # If model_path is not provided, construct it using config_manager
        if model_path is None:
            model_path = str(construct_path("models_dir", "model_{model_name}.pkl", model_name))
        
        super().__init__(model_name, model_path, config or {})
        self.model: Optional[GaussianMixture] = None
        self.trained_columns: List[str] = []

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepares data by selecting columns based on config.
        If no columns specified, auto-selects all numeric columns.
        """
        columns_to_use = self.config.get("gmm_columns_to_use")
        
        if not columns_to_use:
            # Auto-select all numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude common non-feature columns
            exclude_patterns = ['id', 'index', '_id', 'true_state', 'label', 'target']
            for pattern in exclude_patterns:
                numeric_columns = [col for col in numeric_columns if pattern.lower() not in col.lower()]
            
            if not numeric_columns:
                raise ValueError("No numeric columns found in data. Please specify 'gmm_columns_to_use' explicitly.")
            
            columns_to_use = numeric_columns
            log = self._get_logger()
            log.info(f"📋 No columns specified. Auto-selected numeric columns: {columns_to_use}")
        
        # Ensure all specified columns exist in the DataFrame
        missing_cols = [col for col in columns_to_use if col not in data.columns]
        if missing_cols:
            raise ValueError(f"The following columns specified in 'gmm_columns_to_use' are not in the data: {missing_cols}")

        return data[columns_to_use].values, columns_to_use

    def _get_logger(self):
        """Get appropriate logger based on context"""
        try:
            from prefect.logging import get_run_logger
            return get_run_logger()
        except:
            # Fallback to standard logger if not in Prefect context
            return logger

    def train(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """
        Train the GMM model.

        Args:
            data (pd.DataFrame): Input data for training.
            **kwargs: GMM-specific parameters (n_components, covariance_type, etc.).
        """
        log = self._get_logger()
        log.info(f"🏋️ Training GMM model '{self.model_name}'...")
        
        processed_data, self.trained_columns = self._prepare_data(data)

        n_components = self.config.get("gmm_n_components", 3)
        covariance_type = self.config.get("gmm_covariance_type", "diag")
        max_iter = self.config.get("gmm_max_iter", 100)
        random_state = self.config.get("gmm_random_state", 42)
        tol = self.config.get("gmm_tol", 1e-3)
        init_params = self.config.get("gmm_init_params", "kmeans")

        log.debug(f"GMM parameters: n_components={n_components}, covariance_type={covariance_type}, max_iter={max_iter}, random_state={random_state}")
        log.debug(f"Training on columns: {self.trained_columns}")
        log.debug(f"Training data shape: {processed_data.shape}")

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
            tol=tol,
            init_params=init_params,
            **kwargs # Allow overriding from direct call if needed
        )
        
        try:
            self.model.fit(processed_data)
            log.info(f"✅ GMM model '{self.model_name}' trained successfully with {n_components} components")
            log.debug(f"Model converged: {self.model.converged_}")
            log.debug(f"Final log likelihood: {self.model.score(processed_data):.4f}")
        except Exception as e:
            log.error(f"❌ Failed to train GMM model '{self.model_name}': {str(e)}")
            raise

    def sample(self, n_samples: int, **kwargs: Any) -> pd.DataFrame:
        """
        Generate synthetic samples from the trained GMM.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            pd.DataFrame: DataFrame with synthetic data.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded. Call train() or load_model() first.")
        
        log = self._get_logger()
        log.info(f"🧠 Generating {n_samples} samples using GMM model '{self.model_name}'...")
        
        try:
            synthetic_data, _ = self.model.sample(n_samples)
            df_synthetic = pd.DataFrame(synthetic_data, columns=self.trained_columns)
            log.info(f"✅ {n_samples} samples generated successfully")
            log.debug(f"Generated data shape: {df_synthetic.shape}")
            return df_synthetic
        except Exception as e:
            log.error(f"❌ Failed to generate samples with GMM model '{self.model_name}': {str(e)}")
            raise

    def save_model(self) -> str:
        """
        Saves the trained GMM model and the columns it was trained on.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        log = self._get_logger()
        # Ensure the directory exists (construct_path already handles this)
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.model_path, "wb") as f:
                pickle.dump((self.model, self.trained_columns), f)
            log.info(f"📦 GMM Model '{self.model_name}' and column names saved to '{self.model_path}'")
            log.debug(f"Model saved with columns: {self.trained_columns}")
            return self.model_path
        except Exception as e:
            log.error(f"❌ Failed to save GMM model '{self.model_name}' to '{self.model_path}': {str(e)}")
            raise

    def load_model(self) -> None:
        """
        Loads a GMM model and its trained columns from the specified path.
        """
        log = self._get_logger()
        model_file = Path(self.model_path)
        if not model_file.exists():
            error_msg = f"Model file not found at '{self.model_path}'"
            log.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        try:
            with open(model_file, "rb") as f:
                self.model, self.trained_columns = pickle.load(f)
            log.info(f"💡 GMM Model '{self.model_name}' and column names loaded from '{self.model_path}'")
            log.debug(f"Model loaded with columns: {self.trained_columns}")
        except Exception as e:
            log.error(f"❌ Failed to load GMM model from '{self.model_path}': {str(e)}")
            raise