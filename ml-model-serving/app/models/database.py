from sqlalchemy import Column, String, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base, BaseModel

class PredictionRecord(Base, BaseModel):
    __tablename__ = "prediction_records"
    
    request_id = Column(String, unique=True, index=True)
    model_version = Column(String, index=True)
    features = Column(JSON)  # Store as JSON for flexibility
    predictions = Column(JSON)
    inference_time = Column(String)
    status = Column(String, default="success")  # success, error

class ModelMetadata(Base, BaseModel):
    __tablename__ = "model_metadata"
    
    version = Column(String, unique=True, index=True)
    metadata = Column(JSON)  # Full model metadata
    is_active = Column(Boolean, default=True)
    storage_path = Column(String)

class ErrorLog(Base, BaseModel):
    __tablename__ = "error_logs"
    
    model_version = Column(String, index=True)
    error_type = Column(String)
    error_message = Column(String)
    request_data = Column(JSON)
    stack_trace = Column(String)