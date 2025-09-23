from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    async def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            # Classification report
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            self.metrics = {
                "accuracy": float(accuracy),
                "roc_auc": float(roc_auc),
                "classification_report": clf_report,
                "confusion_matrix": cm.tolist(),
                "class_names": list(map(str, np.unique(y_test)))
            }
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def save_evaluation_report(self, file_path: str):
        """Save evaluation report to file"""
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Evaluation report saved to {file_path}")