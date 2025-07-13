from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router as api_router
import joblib
from contextlib import asynccontextmanager
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup: Check if model exists
    if not settings.MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Please run 'python -m app.services.train_model' first"
        )

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    model_info = {}
    try:
        feature_info = joblib.load(settings.FEATURE_INFO_PATH)
        model_info = {
            "n_features": feature_info["n_features"],
            "feature_names": feature_info["feature_names"],
        }
    except (FileNotFoundError, KeyError, joblib.JoblibError) as e:
        model_info = {"status": "Model info not available", "error": str(e)}

    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "documentation": "/docs",
        "model_info": model_info,
        "endpoints": {
            "predict": f"{settings.API_V1_STR}/predict",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {"error": exc.detail, "status_code": exc.status_code}
