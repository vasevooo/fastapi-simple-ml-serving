import numpy as np
from typing import List
import joblib
from pathlib import Path


class DataPreprocessor:
    def __init__(self):
        # Load feature information
        feature_info_path = Path("models/feature_info.joblib")
        if not feature_info_path.exists():
            raise FileNotFoundError(
                "Feature information file not found. Please train the model first."
            )

        self.feature_info = joblib.load(feature_info_path)

    def preprocess(self, features: List[float]) -> np.ndarray:
        """
        Preprocess the input features for the linear regression model.
        Args:
            features: List of input features (must be 4 numerical values)
        Returns:
            Preprocessed features as numpy array
        """
        # Validate number of features
        if len(features) != self.feature_info["n_features"]:
            raise ValueError(
                f"Expected {self.feature_info['n_features']} features, got {len(features)}"
            )

        # Convert to numpy array and reshape for model input
        features_array = np.array(features).reshape(1, -1)

        return features_array
