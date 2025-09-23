import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from prometheus_client import Counter, Histogram, Gauge

from app.utils.logger import logger

class ModelMonitor:
    def __init__(self):
        # Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total predictions made',
            ['model_version', 'status']
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_version']
        )
        
        self.prediction_errors = Counter(
            'model_prediction_errors_total',
            'Total prediction errors',
            ['model_version', 'error_type']
        )
        
        self.model_throughput = Gauge(
            'model_throughput_predictions_per_second',
            'Predictions per second',
            ['model_version']
        )
        
        # In-memory storage for monitoring data
        self.prediction_history: Dict[str, List] = {}
        self.error_history: Dict[str, List] = {}
        self.throughput_data: Dict[str, List] = {}
    
    async def record_prediction(
        self,
        version: str,
        features: List[Any],
        predictions: List[Any],
        inference_time: float,
        request_id: Optional[str] = None
    ) -> None:
        """Record prediction metrics and history"""
        try:
            # Update Prometheus metrics
            self.prediction_counter.labels(version, 'success').inc()
            self.prediction_latency.labels(version).observe(inference_time)
            
            # Update throughput
            current_time = time.time()
            if version not in self.throughput_data:
                self.throughput_data[version] = []
            
            self.throughput_data[version].append(current_time)
            self._clean_throughput_data(version)
            
            throughput = self._calculate_throughput(version)
            self.model_throughput.labels(version).set(throughput)
            
            # Store prediction history
            prediction_record = {
                'timestamp': datetime.now(),
                'request_id': request_id,
                'features': features,
                'predictions': predictions,
                'inference_time': inference_time,
                'version': version
            }
            
            if version not in self.prediction_history:
                self.prediction_history[version] = []
            
            self.prediction_history[version].append(prediction_record)
            
            # Keep only last 1000 predictions for memory efficiency
            if len(self.prediction_history[version]) > 1000:
                self.prediction_history[version] = self.prediction_history[version][-1000:]
                
        except Exception as e:
            logger.error(f"Failed to record prediction: {str(e)}")
    
    async def record_error(self, version: str, error_message: str) -> None:
        """Record prediction error"""
        try:
            self.prediction_counter.labels(version, 'error').inc()
            self.prediction_errors.labels(version, 'prediction_error').inc()
            
            error_record = {
                'timestamp': datetime.now(),
                'version': version,
                'error_message': error_message
            }
            
            if version not in self.error_history:
                self.error_history[version] = []
            
            self.error_history[version].append(error_record)
            
            if len(self.error_history[version]) > 1000:
                self.error_history[version] = self.error_history[version][-1000:]
                
        except Exception as e:
            logger.error(f"Failed to record error: {str(e)}")
    
    async def get_model_stats(self, version: str) -> Dict[str, Any]:
        """Get statistics for a model version"""
        try:
            stats = {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'average_latency': 0.0,
                'throughput': 0.0,
                'recent_errors': []
            }
            
            # Count predictions from Prometheus or history
            if version in self.prediction_history:
                stats['total_predictions'] = len(self.prediction_history[version])
                stats['successful_predictions'] = len(self.prediction_history[version])
                
                if self.prediction_history[version]:
                    latencies = [p['inference_time'] for p in self.prediction_history[version]]
                    stats['average_latency'] = sum(latencies) / len(latencies)
            
            if version in self.error_history:
                stats['failed_predictions'] = len(self.error_history[version])
                stats['recent_errors'] = self.error_history[version][-10:]  # Last 10 errors
            
            # Calculate throughput
            stats['throughput'] = self._calculate_throughput(version)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get model stats: {str(e)}")
            return {}
    
    def _clean_throughput_data(self, version: str) -> None:
        """Clean old throughput data"""
        if version in self.throughput_data:
            current_time = time.time()
            # Keep only data from the last minute
            self.throughput_data[version] = [
                t for t in self.throughput_data[version] 
                if current_time - t <= 60
            ]
    
    def _calculate_throughput(self, version: str) -> float:
        """Calculate predictions per second"""
        if version not in self.throughput_data or not self.throughput_data[version]:
            return 0.0
        
        timestamps = self.throughput_data[version]
        if len(timestamps) <= 1:
            return 0.0
        
        time_range = max(timestamps) - min(timestamps)
        if time_range <= 0:
            return 0.0
        
        return len(timestamps) / time_range
    
    async def check_data_drift(self, version: str, window_size: int = 100) -> Dict[str, Any]:
        """Check for data drift in recent predictions"""
        try:
            if version not in self.prediction_history or len(self.prediction_history[version]) < window_size:
                return {'drift_detected': False, 'confidence': 0.0}
            
            recent_data = self.prediction_history[version][-window_size:]
            features = [pred['features'] for pred in recent_data]
            
            # Simple drift detection based on feature statistics
            # This would be replaced with more sophisticated methods like KS-test, etc.
            feature_array = np.array(features)
            mean_changes = np.std(feature_array, axis=0) / np.mean(feature_array, axis=0)
            
            drift_detected = np.any(mean_changes > 0.1)  # Threshold for drift
            
            return {
                'drift_detected': bool(drift_detected),
                'confidence': float(np.mean(mean_changes)),
                'feature_changes': mean_changes.tolist()
            }
            
        except Exception as e:
            logger.error(f"Failed to check data drift: {str(e)}")
            return {'drift_detected': False, 'confidence': 0.0}