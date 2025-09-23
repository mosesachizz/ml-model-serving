import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.api.endpoints import predictions, models, monitoring, health
from app.ml.model_manager import ModelManager
from app.utils.logger import setup_logging
from app.db.session import AsyncSessionLocal, init_db

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up ML Model Serving API")
    
    # Initialize database
    await init_db()
    
    # Initialize model manager
    model_manager = ModelManager()
    await model_manager.load_models()
    
    # Store model manager in app state
    app.state.model_manager = model_manager
    
    logger.info("Startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down")
    await AsyncSessionLocal.close_all()

app = FastAPI(
    title="ML Model Serving API",
    description="A production-ready machine learning model serving API with MLOps capabilities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
app.include_router(health.router, prefix="/health", tags=["health"])

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )