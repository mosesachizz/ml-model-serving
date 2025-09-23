#!/usr/bin/env python3
"""
Script to monitor data drift and model performance
"""

import requests
import json
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_drift(model_version, api_url="http://localhost:8000"):
    """Check for data drift in model predictions"""
    try:
        response = requests.get(
            f"{api_url}/api/v1/monitoring/models/{model_version}/drift",
            headers={'X-API-Key': 'dev-key-123'}
        )
        
        if response.status_code == 200:
            drift_data = response.json()
            logger.info(f"Drift check for {model_version}: {json.dumps(drift_data, indent=2)}")
            return drift_data
        else:
            logger.error(f"Drift check failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        return None

def get_model_stats(model_version, api_url="http://localhost:8000"):
    """Get model performance statistics"""
    try:
        response = requests.get(
            f"{api_url}/api/v1/monitoring/models/{model_version}/stats",
            headers={'X-API-Key': 'dev-key-123'}
        )
        
        if response.status_code == 200:
            stats = response.json()
            logger.info(f"Model stats for {model_version}: {json.dumps(stats, indent=2)}")
            return stats
        else:
            logger.error(f"Stats check failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Stats check error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Monitor Model Drift and Performance')
    parser.add_argument('version', type=str, help='Model version to monitor')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', help='API URL')
    parser.add_argument('--check-drift', action='store_true', help='Check for data drift')
    parser.add_argument('--check-stats', action='store_true', help='Check model statistics')
    
    args = parser.parse_args()
    
    if args.check_drift:
        check_drift(args.version, args.api_url)
    
    if args.check_stats:
        get_model_stats(args.version, args.api_url)

if __name__ == "__main__":
    main()