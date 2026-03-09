# scripts/evaluate_models.py

import numpy as np
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_autoencoder(reconstruction_error):
    """
    Evaluate autoencoder performance
    
    Args:
        reconstruction_error (np.ndarray): Per-sample reconstruction errors
    
    Returns:
        float: Mean reconstruction error
    """
    mean_error = np.mean(reconstruction_error)
    std_error = np.std(reconstruction_error)
    
    logger.info(f"Autoencoder Evaluation:")
    logger.info(f"  Mean Reconstruction Error: {mean_error:.6f}")
    logger.info(f"  Std Reconstruction Error: {std_error:.6f}")
    logger.info(f"  Min: {np.min(reconstruction_error):.6f}, Max: {np.max(reconstruction_error):.6f}")
    
    return mean_error


def evaluate_clustering(latent_features, cluster_labels):
    """
    Evaluate clustering quality using Silhouette Score
    
    Args:
        latent_features (np.ndarray): Latent features from encoder
        cluster_labels (np.ndarray): Cluster assignments
    
    Returns:
        float: Silhouette score (-1 to 1, higher is better)
    """
    sil_score = silhouette_score(latent_features, cluster_labels)
    
    logger.info(f"KMeans Clustering Evaluation:")
    logger.info(f"  Silhouette Score: {sil_score:.4f}")
    if sil_score < 0:
        logger.warning("  ⚠ Low silhouette score - consider adjusting n_clusters")
    elif sil_score > 0.5:
        logger.info("  ✓ Good cluster separation")
    
    return sil_score


def evaluate_isolation_forest(model, X):
    """
    Evaluate Isolation Forest anomaly detection
    
    Args:
        model: Trained Isolation Forest model
        X (np.ndarray): Feature matrix
    
    Returns:
        float: Contamination rate (proportion of anomalies)
    """
    predictions = model.predict(X)
    contamination = np.sum(predictions == -1) / len(predictions)
    
    logger.info(f"Isolation Forest Evaluation:")
    logger.info(f"  Anomaly Contamination Rate: {contamination:.4f} ({100*contamination:.2f}%)")
    
    return contamination


def evaluate_all(reconstruction_error, latent_features, cluster_labels, isolation_forest, X):
    """
    Evaluate all models and return metrics dictionary
    
    Args:
        reconstruction_error (np.ndarray): From autoencoder
        latent_features (np.ndarray): From encoder
        cluster_labels (np.ndarray): From KMeans
        isolation_forest: Trained model
        X (np.ndarray): Original features
    
    Returns:
        dict: All evaluation metrics
    """
    logger.info("="*70)
    logger.info("EVALUATING ALL MODELS")
    logger.info("="*70)
    
    metrics = {}
    metrics['reconstruction_error_mean'] = evaluate_autoencoder(reconstruction_error)
    metrics['silhouette_score'] = evaluate_clustering(latent_features, cluster_labels)
    metrics['anomaly_contamination_rate'] = evaluate_isolation_forest(isolation_forest, X)
    
    logger.info("="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*70)
    
    return metrics