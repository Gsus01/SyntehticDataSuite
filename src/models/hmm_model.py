import pandas as pd
import numpy as np
import pickle
from hmmlearn import hmm
from pathlib import Path
from typing import List, Tuple, Any, Optional, Union
import logging

from src.models.base import SyntheticModel

# Set up logger for non-Prefect contexts
logger = logging.getLogger(__name__)

class HMMWrapper(SyntheticModel):
    """
    Wrapper for hmmlearn's GaussianHMM model for time series synthetic data generation.
    """
    def __init__(self, model_name: str, model_path: str, config: dict):
        super().__init__(model_name, model_path, config)
        self.model: Optional[hmm.GaussianHMM] = None
        self.trained_columns: List[str] = []
        self.sequence_column: Optional[str] = None

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], Optional[str]]:
        """
        Prepares data by selecting columns based on config.
        For HMM, we need to handle potential sequence/time ordering.
        If no columns are specified, uses all numeric columns except sequence column.
        """
        columns_to_use = self.config.get("hmm_columns_to_use")
        sequence_column = self.config.get("hmm_sequence_column")
        
        # If no columns specified, use all numeric columns
        if not columns_to_use:
            # Get all numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude sequence column if specified
            if sequence_column and sequence_column in numeric_columns:
                numeric_columns.remove(sequence_column)
            
            # Exclude common non-feature columns
            exclude_patterns = ['id', 'index', '_id', 'true_state', 'label', 'target']
            for pattern in exclude_patterns:
                numeric_columns = [col for col in numeric_columns if pattern.lower() not in col.lower()]
            
            if not numeric_columns:
                raise ValueError("No numeric columns found in data. Please specify 'hmm_columns_to_use' explicitly.")
            
            columns_to_use = numeric_columns
            log = self._get_logger()
            log.info(f"📋 No columns specified. Auto-selected numeric columns: {columns_to_use}")
        
        # Ensure all specified columns exist in the DataFrame
        missing_cols = [col for col in columns_to_use if col not in data.columns]
        if missing_cols:
            raise ValueError(f"The following columns specified in 'hmm_columns_to_use' are not in the data: {missing_cols}")

        # Optional: handle sequence/time column for proper ordering
        if sequence_column:
            if sequence_column not in data.columns:
                raise ValueError(f"Sequence column '{sequence_column}' not found in data.")
            # Sort by sequence column to ensure proper temporal order
            data = data.sort_values(by=sequence_column)

        return data[columns_to_use].values, columns_to_use, sequence_column

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
        Train the HMM model.

        Args:
            data (pd.DataFrame): Input data for training.
            **kwargs: HMM-specific parameters that can override config.
        """
        log = self._get_logger()
        log.info(f"🏋️ Training HMM model '{self.model_name}'...")
        
        processed_data, self.trained_columns, self.sequence_column = self._prepare_data(data)

        # HMM hyperparameters with sensible defaults
        n_components = self.config.get("hmm_n_states", 2)
        covariance_type = self.config.get("hmm_covariance_type", "diag")
        n_iter = self.config.get("hmm_n_iter", 100)
        random_state = self.config.get("hmm_random_state", 42)
        tol = self.config.get("hmm_tol", 1e-2)
        algorithm = self.config.get("hmm_algorithm", "viterbi")

        # Validate covariance_type
        valid_cov_types = ["diag", "full", "tied", "spherical"]
        if covariance_type not in valid_cov_types:
            raise ValueError(f"Invalid covariance_type '{covariance_type}'. Must be one of {valid_cov_types}")

        log.debug(f"HMM parameters: n_components={n_components}, covariance_type={covariance_type}, n_iter={n_iter}")
        log.debug(f"Training on columns: {self.trained_columns}")
        log.debug(f"Training data shape: {processed_data.shape}")
        if self.sequence_column:
            log.debug(f"Data ordered by sequence column: {self.sequence_column}")

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
            algorithm=algorithm,
            **kwargs  # Allow overriding from direct call if needed
        )
        
        try:
            self.model.fit(processed_data)
            log.info(f"✅ HMM model '{self.model_name}' trained successfully with {n_components} hidden states")
            log.debug(f"Model converged: {self.model.monitor_.converged}")
            log.debug(f"Final log likelihood: {self.model.score(processed_data):.4f}")
        except Exception as e:
            log.error(f"❌ Failed to train HMM model '{self.model_name}': {str(e)}")
            raise

    def sample(self, n_samples: int, **kwargs: Any) -> pd.DataFrame:
        """
        Generate synthetic samples from the trained HMM.

        Args:
            n_samples (int): Number of samples to generate.
            **kwargs: Additional sampling parameters.

        Returns:
            pd.DataFrame: DataFrame with synthetic sequential data.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded. Call train() or load_model() first.")
        
        log = self._get_logger()
        log.info(f"🧠 Generating {n_samples} samples using HMM model '{self.model_name}'...")
        
        try:
            # Generate samples and hidden states
            synthetic_data, hidden_states = self.model.sample(n_samples)
            
            # Create DataFrame with synthetic data
            df_synthetic = pd.DataFrame(synthetic_data, columns=self.trained_columns)
            
            # Add hidden states as additional information
            df_synthetic['hidden_state'] = hidden_states
            
            # Add sequence index to maintain temporal order
            df_synthetic['sequence_index'] = range(len(df_synthetic))
            
            log.info(f"✅ {n_samples} sequential samples generated successfully")
            log.debug(f"Generated data shape: {df_synthetic.shape}")
            log.debug(f"Number of unique hidden states in sample: {len(np.unique(hidden_states))}")
            
            return df_synthetic
        except Exception as e:
            log.error(f"❌ Failed to generate samples with HMM model '{self.model_name}': {str(e)}")
            raise

    def predict_hidden_states(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict the most likely sequence of hidden states for given observations.
        
        Args:
            data (pd.DataFrame): Input observations.
            
        Returns:
            np.ndarray: Predicted hidden states.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded. Call train() or load_model() first.")
            
        log = self._get_logger()
        
        # Prepare data using the same columns as training
        if not all(col in data.columns for col in self.trained_columns):
            missing_cols = [col for col in self.trained_columns if col not in data.columns]
            raise ValueError(f"Missing columns in input data: {missing_cols}")
            
        processed_data = data[self.trained_columns].values
        
        try:
            hidden_states = self.model.predict(processed_data)
            log.debug(f"Predicted hidden states for {len(processed_data)} observations")
            return hidden_states
        except Exception as e:
            log.error(f"❌ Failed to predict hidden states: {str(e)}")
            raise

    def save_model(self) -> str:
        """
        Saves the trained HMM model and metadata.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        log = self._get_logger()
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model, columns, and sequence column info
            model_data = {
                'model': self.model,
                'trained_columns': self.trained_columns,
                'sequence_column': self.sequence_column,
                'config': self.config
            }
            
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
                
            log.info(f"📦 HMM Model '{self.model_name}' and metadata saved to '{self.model_path}'")
            log.debug(f"Model saved with columns: {self.trained_columns}")
            log.debug(f"Sequence column: {self.sequence_column}")
            return self.model_path
        except Exception as e:
            log.error(f"❌ Failed to save HMM model '{self.model_name}' to '{self.model_path}': {str(e)}")
            raise

    def load_model(self) -> None:
        """
        Loads an HMM model and its metadata from the specified path.
        """
        log = self._get_logger()
        model_file = Path(self.model_path)
        if not model_file.exists():
            error_msg = f"Model file not found at '{self.model_path}'"
            log.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.trained_columns = model_data['trained_columns']
            self.sequence_column = model_data.get('sequence_column')
            
            log.info(f"💡 HMM Model '{self.model_name}' and metadata loaded from '{self.model_path}'")
            log.debug(f"Model loaded with columns: {self.trained_columns}")
            log.debug(f"Sequence column: {self.sequence_column}")
        except Exception as e:
            log.error(f"❌ Failed to load HMM model from '{self.model_path}': {str(e)}")
            raise 