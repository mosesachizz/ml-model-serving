#!/usr/bin/env python3
"""
Script to deploy ML models to serving environment
"""

import argparse
import requests
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_model(model_version, api_url="http://localhost:8000"):
    """Deploy a trained model to the serving API"""
    try:
        model_path = Path(f"models/{model_version}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {model_version} not found")
        
        # Load model and metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Read model file
        with open(model_path / "model.joblib", 'rb') as f:
            model_data = f.read()
        
        # Prepare request
        files = {
            'model_file': ('model.joblib', model_data, 'application/octet-stream')
        }
        data = {
            'metadata': json.dumps(metadata)
        }
        
        # Send to API
        response = requests.post(
            f"{api_url}/api/v1/models/{model_version}",
            files=files,
            data=data,
            headers={'X-API-Key': 'dev-key-123'}
        )
        
        if response.status_code == 201:
            logger.info(f"Model {model_version} deployed successfully")
            return True
        else:
            logger.error(f"Deployment failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Deploy ML Model')
    parser.add_argument('version', type=str, help='Model version to deploy')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', help='API URL')
    
    args = parser.parse_args()
    
    success = deploy_model(args.version, args.api_url)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()