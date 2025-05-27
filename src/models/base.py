from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd

class SyntheticModel(ABC):
    """
    Abstract base class for synthetic data generation models.
    """
    def __init__(self, model_name: str, model_path: str, config: dict):
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.model: Any = None # To store the trained model object
        self.trained_columns: List[str] = []

    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """
        Train the model.
        
        Args:
            data (pd.DataFrame): The input data for training.
            **kwargs: Additional model-specific training parameters.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int, **kwargs: Any) -> pd.DataFrame:
        """
        Generate synthetic samples from the trained model.

        Args:
            n_samples (int): The number of samples to generate.
            **kwargs: Additional model-specific sampling parameters.

        Returns:
            pd.DataFrame: A DataFrame containing the synthetic samples.
        """
        pass

    @abstractmethod
    def save_model(self) -> str:
        """
        Save the trained model to a file.

        Returns:
            str: The path where the model was saved.
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load a pre-trained model from a file.
        """
        pass 