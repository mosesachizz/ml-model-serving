from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import logging
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_params = None
    
    async def train_model(self, df: pd.DataFrame, target_column: str) -> dict:
        """Train machine learning model"""
        try:
            # Split data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            # Evaluate model
            y_pred = self.best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model training completed. Best accuracy: {accuracy:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            return {
                "model": self.best_model,
                "accuracy": accuracy,
                "best_params": self.best_params,
                "feature_importance": dict(zip(X.columns, self.best_model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise