#!/usr/bin/env python3
"""
Script to train and deploy ML models
"""

import asyncio
import logging
import argparse
from training_pipeline.pipeline import TrainingPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description='Train ML Model')
    parser.add_argument('--data-path', type=str, default='data/processed/iris_processed.csv', help='Path to training data')
    parser.add_argument('--target-column', type=str, default='target', help='Target column name')
    parser.add_argument('--version', type=str, help='Model version (default: auto-generated)')
    
    args = parser.parse_args()
    
    try:
        pipeline = TrainingPipeline()
        version = await pipeline.run()
        logger.info(f"Training completed successfully. Model version: {version}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))