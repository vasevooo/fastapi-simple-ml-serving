from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path


def train_and_save_model():
    # Create a synthetic dataset for demonstration
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)

    # Create and train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save the model
    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save feature information
    feature_info = {
        "n_features": 4,
        "feature_names": [f"feature_{i + 1}" for i in range(4)],
    }
    joblib.dump(feature_info, models_dir / "feature_info.joblib")
    print("Feature information saved")


if __name__ == "__main__":
    train_and_save_model()
