import logging
import aiofiles
import aiohttp
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import tempfile

from app.core.config import settings
from app.utils.logger import logger

class ModelStorage:
    def __init__(self):
        self.storage_type = settings.MODEL_STORAGE_TYPE
        self.base_path = Path(settings.M