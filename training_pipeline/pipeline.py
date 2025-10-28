import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from training_pipeline.data_processing import DataProcessor
from training_pipeline.model_training import ModelTrainer
from training_pipeline.model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
    
    async def run(self):
        """Run the complete training pipeline"""
        try:
            logger.info("Starting training pipeline")
            
            # 1. Data processing
            logger.info("Step 1: Processing data")
            processed_data = await self.data_processor.process_data()
            
            # 2. Model training
            logger.info("Step 2: Training model")
            model, training_metrics = await self.model_trainer.train_model(processed_data)
            
            # 3. Model evaluation
            logger.info("Step 3: Evaluating model")
            evaluation_metrics = await self.model_evaluator.evaluate_model(model, processed_data)
            
            # 4. Save model and metadata
            logger.info("Step 4: Saving model")
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self._save_model(model, training_metrics, evaluation_metrics, version)
            
            logger.info(f"Training pipeline completed successfully. Model version: {version}")
            return version
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    async def _save_model(self, model, training_metrics, evaluation_metrics, version):
        """Save model and metadata"""
        # This would integrate with your ModelManager
        metadata = {
            "version": version,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics,
            "created_at": datetime.now().isoformat(),
            "preprocessing": self.data_processor.get_preprocessing_config()
        }
        
        # Save model using your storage system
        # await model_manager.update_model(version, model, metadata)
        logger.info(f"Model {version} saved with metadata: {metadata}")

async def main():
    """Main pipeline execution"""
    pipeline = TrainingPipeline()
    try:
        version = await pipeline.run()
        print(f"Pipeline completed. Model version: {version}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    asyncio.run(main())