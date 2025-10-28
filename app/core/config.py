import os
from typing import List, Optional
from pydantic import BaseSettings, AnyUrl, validator
from functools import lru_cache

class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ml_serving"
    
    # Model Storage
    MODEL_STORAGE_TYPE: str = "local"  # local, s3, gcs
    MODEL_STORAGE_PATH: str = "./models"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: Optional[str] = None
    
    # Monitoring
    PROMETHEUS_URL: str = "http://localhost:9090"
    GRAFANA_URL: str = "http://localhost:3000"
    ENABLE_METRICS: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = ["dev-key-123"]
    
    # Model Configuration
    DEFAULT_MODEL_VERSION: str = "v1"
    MODEL_LOAD_TIMEOUT: int = 30
    MAX_PREDICTION_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @validator("API_KEYS", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()