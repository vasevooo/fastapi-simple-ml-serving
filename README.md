# FastAPI ML Project

Simple ML model serving API built with FastAPI.

## Quick Start

```bash
# Start the service
docker compose up --build
```

The API will be available at `http://localhost:8000`

## Usage

Send prediction request:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, -0.2, 1.3, -0.8]}'
```

Expected response:
```json
{
    "prediction": 1.234  # Predicted value from the model
}
```

API documentation available at:
- http://localhost:8000/docs
