import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from pathlib import Path
from typing import List, Tuple, Any, Optional, Union

from src.models.base import SyntheticModel

class GMMWrapper(SyntheticModel):
    """
    Wrapper for scikit-learn's GaussianMixture model.
    """
    def __init__(self, model_name: str, model_path: str, config: dict):
        super().__init__(model_name, model_path, config)
        self.model: Optional[GaussianMixture] = None
        self.trained_columns: List[str] = []

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepares data by selecting columns based on config.
        """
        columns_to_use = self.config.get("gmm_columns_to_use")
        if not columns_to_use:
            raise ValueError("GMM configuration must specify 'gmm_columns_to_use' as a list of column names.")
        
        # Ensure all specified columns exist in the DataFrame
        missing_cols = [col for col in columns_to_use if col not in data.columns]
        if missing_cols:
            raise ValueError(f"The following columns specified in 'gmm_columns_to_use' are not in the data: {missing_cols}")

        return data[columns_to_use].values, columns_to_use

    def train(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """
        Train the GMM model.

        Args:
            data (pd.DataFrame): Input data for training.
            **kwargs: GMM-specific parameters (n_components, covariance_type, etc.).
        """
        print(f"🏋️ Training GMM model '{self.model_name}'...")
        
        processed_data, self.trained_columns = self._prepare_data(data)

        n_components = self.config.get("gmm_n_components", 3)
        covariance_type = self.config.get("gmm_covariance_type", "diag")
        max_iter = self.config.get("gmm_max_iter", 100)
        random_state = self.config.get("gmm_random_state", 42)

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs # Allow overriding from direct call if needed
        )
        self.model.fit(processed_data)
        print(f"✅ GMM model '{self.model_name}' trained successfully.")

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
        
        print(f"🧠 Generating {n_samples} samples using GMM model '{self.model_name}'...")
        synthetic_data, _ = self.model.sample(n_samples)
        df_synthetic = pd.DataFrame(synthetic_data, columns=self.trained_columns)
        print(f"✅ {n_samples} samples generated.")
        return df_synthetic

    def save_model(self) -> str:
        """
        Saves the trained GMM model and the columns it was trained on.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, self.trained_columns), f)
        print(f"📦 GMM Model '{self.model_name}' and column names saved to '{self.model_path}'.")
        return self.model_path

    def load_model(self) -> None:
        """
        Loads a GMM model and its trained columns from the specified path.
        """
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'")

        with open(model_file, "rb") as f:
            self.model, self.trained_columns = pickle.load(f)
        print(f"💡 GMM Model '{self.model_name}' and column names loaded from '{self.model_path}'.") 