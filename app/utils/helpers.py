import uuid
import hashlib
from typing import Any, Dict
import json
from datetime import datetime

def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())

def format_bytes(size: int) -> str:
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def safe_json_dumps(data: Any) -> str:
    """Safely convert data to JSON string"""
    def default_serializer(obj):
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    return json.dumps(data, default=default_serializer)

def hash_data(data: Any) -> str:
    """Generate hash for data"""
    data_str = safe_json_dumps(data)
    return hashlib.md5(data_str.encode()).hexdigest()

def validate_features(features: list, expected_shape: tuple) -> bool:
    """Validate features shape and type"""
    if not features:
        return False
    
    # Check if features match expected shape
    if hasattr(features[0], '__len__'):
        # 2D array
        if len(features) != expected_shape[0]:
            return False
        if len(features[0]) != expected_shape[1]:
            return False
    else:
        # 1D array
        if len(features) != expected_shape[0]:
            return False
    
    return True