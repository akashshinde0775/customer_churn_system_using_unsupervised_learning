# models/clustering_model.py

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusteringModel:
    """
    Wrapper class for MiniBatch K-Means clustering.
    Compatible with your existing implementation while providing class interface for FastAPI.
    """
    
    def __init__(self, n_clusters=3, random_state=42, batch_size=256, max_iter=100):
        """
        Initialize the clustering model.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed
            batch_size (int): Batch size for MiniBatch
            max_iter (int): Maximum iterations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_iter = max_iter
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=10
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cluster_distance = None
        self.cluster_labels = None
        self.stable_cluster = None
        self.cluster_centers = None
    
    def train(self, latent_features):
        """
        Train MiniBatchKMeans on latent features.
        
        Args:
            latent_features (np.ndarray): Latent features from encoder
            
        Returns:
            dict: Training results
        """
        logger.info("="*70)
        logger.info("TRAINING MINIBATCHKMEANS CLUSTERING")
        logger.info("="*70)
        logger.info(f"Latent features shape: {latent_features.shape}")
        logger.info(f"Number of clusters: {self.n_clusters}")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(latent_features)
        
        # Fit the model
        self.kmeans.fit(X_scaled)
        self.cluster_labels = self.kmeans.predict(X_scaled)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        logger.info("✓ KMeans clustering complete")
        
        # Identify stable cluster (largest cluster)
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        self.stable_cluster = unique[np.argmax(counts)]
        stable_cluster_size = np.max(counts)
        
        logger.info(f"Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} customers ({100*count/len(self.cluster_labels):.1f}%)")
        logger.info(f"✓ Stable cluster: {self.stable_cluster} (size: {stable_cluster_size})")
        
        # Compute cluster distance (distance to nearest cluster center)
        distances_to_centers = np.zeros((len(X_scaled), self.n_clusters))
        for i in range(self.n_clusters):
            distances_to_centers[:, i] = np.linalg.norm(
                X_scaled - self.kmeans.cluster_centers_[i],
                axis=1
            )
        
        self.cluster_distance = np.min(distances_to_centers, axis=1)
        
        self.is_trained = True
        
        logger.info(f"Cluster distance statistics:")
        logger.info(f"  Mean: {np.mean(self.cluster_distance):.6f}")
        logger.info(f"  Min: {np.min(self.cluster_distance):.6f}, Max: {np.max(self.cluster_distance):.6f}")
        
        return {
            'kmeans': self.kmeans,
            'cluster_distance': self.cluster_distance,
            'cluster_labels': self.cluster_labels,
            'stable_cluster': self.stable_cluster,
            'cluster_centers': self.cluster_centers
        }
    
    def get_cluster_distance(self, X):
        """
        Get distance to nearest cluster for each sample.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Normalized distances [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Compute distances to all cluster centers
        distances_to_centers = np.zeros((len(X_scaled), self.n_clusters))
        for i in range(self.n_clusters):
            distances_to_centers[:, i] = np.linalg.norm(
                X_scaled - self.cluster_centers[i],
                axis=1
            )
        
        # Distance to nearest cluster
        distances = np.min(distances_to_centers, axis=1)
        
        # Normalize
        if len(distances) > 0:
            min_val = distances.min()
            max_val = distances.max()
            if max_val > min_val:
                normalized = (distances - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(distances)
        else:
            normalized = np.array([0.0])
        
        return normalized
    
    def predict(self, X):
        """Predict cluster labels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def update(self, X_new):
        """
        Incrementally train with new data.
        
        Args:
            X_new (np.ndarray): New latent features
        """
        logger.info(f"Incremental training with {len(X_new)} new samples...")
        
        X_scaled = self.scaler.transform(X_new)
        self.kmeans.partial_fit(X_scaled)
        
        logger.info("✓ KMeans updated")
    
    def save(self, filepath):
        """Save the trained model to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            joblib.dump({
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'n_clusters': self.n_clusters,
                'is_trained': self.is_trained,
                'cluster_distance': self.cluster_distance,
                'cluster_labels': self.cluster_labels,
                'stable_cluster': self.stable_cluster,
                'cluster_centers': self.cluster_centers
            }, filepath)
            
            logger.info(f"✓ Clustering model saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Error saving clustering model: {e}")
            raise
    
    @staticmethod
    def load(filepath):
        """Load a trained model from disk"""
        try:
            data = joblib.load(filepath)
            
            instance = ClusteringModel(
                n_clusters=data['n_clusters']
            )
            instance.kmeans = data['kmeans']
            instance.scaler = data['scaler']
            instance.is_trained = data['is_trained']
            instance.cluster_distance = data['cluster_distance']
            instance.cluster_labels = data['cluster_labels']
            instance.stable_cluster = data['stable_cluster']
            instance.cluster_centers = data['cluster_centers']
            
            logger.info(f"✓ Clustering model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Error loading clustering model: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the trained model"""
        return {
            "model_type": "MiniBatch K-Means",
            "is_trained": self.is_trained,
            "n_clusters": self.n_clusters,
            "stable_cluster": int(self.stable_cluster) if self.stable_cluster is not None else None
        }