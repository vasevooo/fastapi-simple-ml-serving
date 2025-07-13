from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.predictor import ModelPredictor
from app.services.preprocessor import DataPreprocessor
from typing import List

router = APIRouter()


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,  # ... means required
        description="List of 4 numerical features for prediction",
        min_items=4,
        max_items=4,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, -0.2, 1.3, -0.8]  # Example features
            }
        }


class PredictionOutput(BaseModel):
    prediction: float = Field(..., description="Model prediction value")


@router.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Make prediction using linear regression model",
    description="Predicts a value using a trained linear regression model based on 4 input features",
)
async def predict(input_data: PredictionInput):
    try:
        # Preprocess the input data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(input_data.features)

        # Make prediction
        predictor = ModelPredictor()
        prediction = predictor.predict(processed_data)

        return PredictionOutput(prediction=prediction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
