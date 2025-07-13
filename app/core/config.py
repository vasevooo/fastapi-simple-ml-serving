from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML Model API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for machine learning model predictions"

    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Model Configuration
    MODEL_PATH: Path = Path("models/model.joblib")
    FEATURE_INFO_PATH: Path = Path("models/feature_info.joblib")

    class Config:
        case_sensitive = True


# Create global settings object
settings = Settings()
