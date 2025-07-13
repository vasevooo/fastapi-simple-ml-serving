import numpy as np
import joblib
from pathlib import Path


class ModelPredictor:
    def __init__(self):
        # Load the model
        model_path = Path("models/model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(
                "Model file not found. Please train and save the model first."
            )

        self.model = joblib.load(model_path)

    def predict(self, features: np.ndarray) -> float:
        """
        Make predictions using the loaded model.
        Args:
            features: Preprocessed features as numpy array
        Returns:
            Model prediction
        """
        prediction = self.model.predict(features)
        return float(prediction[0])  # Convert to float for JSON serialization
