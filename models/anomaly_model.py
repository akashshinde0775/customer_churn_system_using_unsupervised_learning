# models/anomaly_model.py

import numpy as np
from sklearn.ensemble import IsolationForest
import logging
import joblib
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Wrapper class for Isolation Forest anomaly detection.
    Compatible with your existing implementation while providing class interface for FastAPI.
    """
    
    def __init__(self, n_estimators=100, warm_start=True, contamination='auto', 
                 random_state=42, max_samples='auto'):
        """
        Initialize the anomaly detector.
        
        Args:
            n_estimators (int): Number of trees
            warm_start (bool): Enable warm start for online learning
            contamination (float or 'auto'): Expected proportion of anomalies
            random_state (int): Random seed
            max_samples (int or 'auto'): Number of samples for each tree
        """
        self.n_estimators = n_estimators
        self.warm_start = warm_start
        self.contamination = contamination
        self.random_state = random_state
        self.max_samples = max_samples
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            warm_start=warm_start,
            contamination=contamination,
            random_state=random_state,
            max_samples=max_samples,
            n_jobs=-1
        )
        
        self.is_trained = False
        self.anomaly_score = None
        self.predictions = None
        self.contamination_rate = None
    
    def train(self, X):
        """
        Train Isolation Forest for anomaly detection
        
        Args:
            X (np.ndarray): Feature matrix
        
        Returns:
            dict: Training results
        """
        logger.info("="*70)
        logger.info("TRAINING ISOLATION FOREST (ONLINE ANOMALY DETECTION)")
        logger.info("="*70)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Configuration: n_estimators={self.n_estimators}, warm_start={self.warm_start}, contamination={self.contamination}")
        
        self.model.fit(X)
        logger.info("✓ Isolation Forest training complete")
        
        # Get anomaly scores
        raw_scores = self.model.decision_function(X)
        self.anomaly_score = -raw_scores  # Convert to positive anomaly score
        
        # Get predictions
        self.predictions = self.model.predict(X)
        n_anomalies = np.sum(self.predictions == -1)
        self.contamination_rate = n_anomalies / len(self.predictions)
        
        self.is_trained = True
        
        logger.info(f"Anomaly detection results:")
        logger.info(f"  Predicted anomalies: {n_anomalies} ({100*self.contamination_rate:.2f}%)")
        logger.info(f"  Anomaly score - Mean: {np.mean(self.anomaly_score):.6f}")
        logger.info(f"  Anomaly score - Min: {np.min(self.anomaly_score):.6f}, Max: {np.max(self.anomaly_score):.6f}")
        
        return {
            'isolation_forest': self.model,
            'anomaly_score': self.anomaly_score,
            'predictions': self.predictions,
            'contamination_rate': self.contamination_rate
        }
    
    def predict(self, X):
        """
        Predict anomaly scores for new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Anomaly scores normalized to [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        raw_scores = self.model.decision_function(X)
        anomaly_scores = -raw_scores
        
        # Normalize to [0, 1]
        if len(anomaly_scores) > 0:
            min_val = anomaly_scores.min()
            max_val = anomaly_scores.max()
            if max_val > min_val:
                anomaly_scores = (anomaly_scores - min_val) / (max_val - min_val)
            else:
                anomaly_scores = np.zeros_like(anomaly_scores)
        
        return anomaly_scores
    
    def update(self, X_new):
        """
        Update with new data (online learning)
        
        Args:
            X_new (np.ndarray): New data to learn from
        """
        logger.info(f"Updating Isolation Forest with {len(X_new)} new samples...")
        self.model.fit(X_new)
        logger.info("✓ Isolation Forest updated")
    
    def save(self, filepath):
        """Save the trained model to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'is_trained': self.is_trained,
                'contamination_rate': self.contamination_rate,
                'n_estimators': self.n_estimators
            }, filepath)
            logger.info(f"✓ Anomaly detector saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Error saving anomaly detector: {e}")
            raise
    
    @staticmethod
    def load(filepath):
        """Load a trained model from disk"""
        try:
            data = joblib.load(filepath)
            
            instance = AnomalyDetector(
                n_estimators=data.get('n_estimators', 100)
            )
            instance.model = data['model']
            instance.is_trained = data['is_trained']
            instance.contamination_rate = data['contamination_rate']
            
            logger.info(f"✓ Anomaly detector loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Error loading anomaly detector: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the trained model"""
        return {
            "model_type": "Isolation Forest",
            "is_trained": self.is_trained,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "contamination_rate": self.contamination_rate
        }