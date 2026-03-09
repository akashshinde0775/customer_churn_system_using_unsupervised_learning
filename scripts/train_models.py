# scripts/train_models.py

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder_model import AutoencoderModel
from models.clustering_model import ClusteringModel
from models.anomaly_model import AnomalyDetector
from scripts.preprocess_data import DataPreprocessor
from scripts.evaluate_models import evaluate_all

import pymysql

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'customer_features_processed.csv')
FETCHED_CSV = os.path.join(BASE_DIR, 'data', 'fetched', 'customer_features_fetched.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'trained_model')

# Model paths for saving
AUTOENCODER_PATH = os.path.join(MODELS_DIR, 'autoencoder_model.pkl')
CLUSTERING_PATH = os.path.join(MODELS_DIR, 'clustering_model.pkl')
ANOMALY_PATH = os.path.join(MODELS_DIR, 'anomaly_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "customer_churn_system",
    "port": 3306
}

# ========== Database Functions ==========

def get_db_connection_wrapper():
    """Get database connection using PyMySQL"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        logger.info("✓ Database connection established")
        return conn
    except pymysql.Error as err:
        logger.error(f"Database connection failed: {err}")
        return None


def insert_model_registry(conn, model_name, training_type, training_data_size):
    """Insert training event in model_registry"""
    cursor = conn.cursor()
    training_date = datetime.now()
    
    sql = """
        INSERT INTO model_registry
        (model_name, training_type, training_date, training_data_size, is_active)
        VALUES (%s, %s, %s, %s, %s)
    """
    data = (model_name, training_type, training_date, training_data_size, True)
    
    try:
        cursor.execute(sql, data)
        conn.commit()
        model_bundle_id = cursor.lastrowid
        logger.info(f"✓ Training event recorded: ID={model_bundle_id}, Type={training_type}, Size={training_data_size}")
        return model_bundle_id
    except pymysql.Error as err:
        logger.error(f"Error inserting model registry: {err}")
        conn.rollback()
        return None
    finally:
        cursor.close()


def insert_model_evaluation(conn, model_bundle_id, metrics):
    """Insert evaluation metrics"""
    cursor = conn.cursor()
    
    sql = """
        INSERT INTO model_evaluation
        (model_bundle_id, reconstruction_error_mean, silhouette_score, anomaly_contamination_rate)
        VALUES (%s, %s, %s, %s)
    """
    data = (
        model_bundle_id,
        float(metrics['reconstruction_error_mean']),
        float(metrics['silhouette_score']),
        float(metrics['anomaly_contamination_rate'])
    )
    
    try:
        cursor.execute(sql, data)
        conn.commit()
        logger.info("✓ Evaluation metrics stored")
    except pymysql.Error as err:
        logger.error(f"Error inserting evaluation metrics: {err}")
        conn.rollback()
    finally:
        cursor.close()


def insert_training_summary(conn, model_bundle_id, total_customers, stable, at_risk, high_risk):
    """Insert training summary"""
    cursor = conn.cursor()
    
    sql = """
        INSERT INTO training_summary
        (model_bundle_id, total_customers_trained, stable_customers, at_risk_customers, high_risk_customers)
        VALUES (%s, %s, %s, %s, %s)
    """
    data = (model_bundle_id, total_customers, stable, at_risk, high_risk)
    
    try:
        cursor.execute(sql, data)
        conn.commit()
        logger.info(f"✓ Training summary stored")
        logger.info(f"  Stable: {stable}, At-Risk: {at_risk}, High-Risk: {high_risk}")
    except pymysql.Error as err:
        logger.error(f"Error inserting training summary: {err}")
        conn.rollback()
    finally:
        cursor.close()


def insert_customer_scores(conn, all_customers):
    """Insert/Update customer risk scores"""
    cursor = conn.cursor()
    
    sql = """
        INSERT INTO customer_risk_scores
        (customer_id, model_bundle_id, cluster_distance, anomaly_score, reconstruction_error, final_risk_score, risk_category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            model_bundle_id = VALUES(model_bundle_id),
            cluster_distance = VALUES(cluster_distance),
            anomaly_score = VALUES(anomaly_score),
            reconstruction_error = VALUES(reconstruction_error),
            final_risk_score = VALUES(final_risk_score),
            risk_category = VALUES(risk_category),
            prediction_time = NOW()
    """
    
    try:
        batch_size = 500
        total_batches = (len(all_customers) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(all_customers), batch_size), 1):
            batch = all_customers[i:i+batch_size]
            cursor.executemany(sql, batch)
            conn.commit()
            logger.info(f"✓ Updated batch {batch_num}/{total_batches}: {i+len(batch)}/{len(all_customers)} scores")
        
        logger.info(f"✓ All {len(all_customers)} customer risk scores updated")
    except pymysql.Error as err:
        logger.error(f"Error inserting customer scores: {err}")
        conn.rollback()
    finally:
        cursor.close()


# ========== Model Management ==========

def load_existing_models():
    """Load existing trained models"""
    try:
        autoencoder = AutoencoderModel.load(AUTOENCODER_PATH)
        clustering = ClusteringModel.load(CLUSTERING_PATH)
        anomaly = AnomalyDetector.load(ANOMALY_PATH)
        preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        
        logger.info("✓ Loaded existing trained models")
        return {
            'autoencoder': autoencoder,
            'clustering': clustering,
            'anomaly': anomaly,
            'preprocessor': preprocessor
        }
    except Exception as e:
        logger.warning(f"Could not load existing models: {e}")
        return None


def save_all_models(autoencoder, clustering, anomaly, preprocessor):
    """Save all trained models"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        autoencoder.save(AUTOENCODER_PATH)
        clustering.save(CLUSTERING_PATH)
        anomaly.save(ANOMALY_PATH)
        preprocessor.save(PREPROCESSOR_PATH)
        logger.info(f"✓ All models saved to {MODELS_DIR}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise


# ========== Score Utilities ==========

def normalize_score(score, min_val=None, max_val=None):
    """Normalize score to 0-1 range"""
    if min_val is None:
        min_val = np.min(score)
    if max_val is None:
        max_val = np.max(score)
    
    if max_val == min_val:
        normalized = np.zeros_like(score, dtype=float)
    else:
        normalized = (score - min_val) / (max_val - min_val)
    
    return np.clip(normalized, 0, 1)


def fuse_scores(cluster_dist, anomaly_score, recon_error):
    """Fuse component scores into final risk score (0-1 range)"""
    logger.info("Fusing component scores...")
    
    # Normalize each component to 0-1 range
    cluster_norm = normalize_score(cluster_dist)
    anomaly_norm = normalize_score(anomaly_score)
    recon_norm = normalize_score(recon_error)
    
    # Average the normalized scores
    final_score = (cluster_norm + anomaly_norm + recon_norm) / 3.0
    
    logger.info(f"  Mean final score: {np.mean(final_score):.4f}")
    logger.info(f"  Min: {np.min(final_score):.4f}, Max: {np.max(final_score):.4f}")
    
    return final_score


def categorize_risk(final_score):
    """Categorize risk scores into categories"""
    risk_category = np.where(
        final_score < 0.33,
        'Stable',
        np.where(
            final_score < 0.67,
            'At Risk',
            'High Risk'
        )
    )
    logger.info("Risk categorization complete")
    return risk_category


# ========== Main Training Pipeline ==========

def main():
    """Main training pipeline - handles both initial and incremental training"""
    
    logger.info("="*70)
    logger.info("CUSTOMER CHURN RISK ANALYSIS - TRAINING PIPELINE")
    logger.info("="*70)
    
    start_time = time.time()
    
    # --- Load data ---
    logger.info("\n[STEP 1] Loading preprocessed features...")
    
    if not os.path.exists(PROCESSED_CSV):
        logger.error(f"Processed data not found at {PROCESSED_CSV}")
        return
    
    df_processed = pd.read_csv(PROCESSED_CSV)
    X = df_processed.values
    logger.info(f"✓ Loaded {len(X)} customer samples with {X.shape[1]} features")
    
    df_fetched = pd.read_csv(FETCHED_CSV)
    customer_ids = df_fetched['customer_id'].values
    logger.info(f"✓ Loaded {len(customer_ids)} customer IDs")
    
    # --- Initialize or load models ---
    logger.info("\n[STEP 2] Initializing models...")
    
    existing_models = load_existing_models()
    
    if existing_models is None:
        # INITIAL TRAINING
        logger.info("\n>>> INITIAL TRAINING MODE <<<")
        
        # Initialize preprocessor
        logger.info("\n[STEP 2a] Initializing preprocessor...")
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        logger.info(f"✓ Data preprocessed and scaled")
        
        # Train autoencoder
        logger.info("\n[STEP 2b] Training Autoencoder...")
        autoencoder = AutoencoderModel(input_dim=X.shape[1], encoding_dim=6, hidden_dim=8, epochs=50)
        ae_results = autoencoder.train(X_scaled, validation_split=0.1, batch_size=32, verbose=1)
        reconstruction_error = ae_results['reconstruction_error']
        logger.info(f"✓ Reconstruction error - Mean: {np.mean(reconstruction_error):.6f}")
        
        # Extract latent features
        logger.info("\n[STEP 2c] Extracting latent features...")
        latent_features = autoencoder.get_encoding(X_scaled)
        logger.info(f"✓ Latent features shape: {latent_features.shape}")
        
        # Train clustering model
        logger.info("\n[STEP 2d] Training Clustering Model...")
        clustering = ClusteringModel(n_clusters=5, random_state=42)
        clustering_results = clustering.train(latent_features)
        cluster_distance = clustering_results['cluster_distance']
        cluster_labels = clustering_results['cluster_labels']
        logger.info(f"✓ Cluster distance - Mean: {np.mean(cluster_distance):.6f}")
        
        # Train anomaly detector
        logger.info("\n[STEP 2e] Training Anomaly Detector...")
        anomaly = AnomalyDetector(n_estimators=100, contamination='auto')
        anomaly_results = anomaly.train(X_scaled)
        anomaly_score = anomaly_results['anomaly_score']
        logger.info(f"✓ Anomaly score - Mean: {np.mean(anomaly_score):.6f}")
        
        training_type = "initial"
    
    else:
        # INCREMENTAL UPDATE
        logger.info("\n>>> INCREMENTAL TRAINING MODE <<<")
        
        autoencoder = existing_models['autoencoder']
        clustering = existing_models['clustering']
        anomaly = existing_models['anomaly']
        preprocessor = existing_models['preprocessor']
        
        logger.info("\n[STEP 2a] Preprocessing new data...")
        X_scaled = preprocessor.transform(X)
        logger.info(f"✓ Data preprocessed")
        
        logger.info("\n[STEP 2b] Incrementally updating Autoencoder...")
        autoencoder.train(X_scaled, validation_split=0.1, batch_size=32, verbose=0)
        reconstruction_error = autoencoder.predict(X_scaled)
        logger.info(f"✓ Autoencoder updated - Error mean: {np.mean(reconstruction_error):.6f}")
        
        logger.info("\n[STEP 2c] Extracting latent features...")
        latent_features = autoencoder.get_encoding(X_scaled)
        logger.info(f"✓ Latent features shape: {latent_features.shape}")
        
        logger.info("\n[STEP 2d] Incrementally updating Clustering Model...")
        clustering.update(latent_features)
        cluster_distance = clustering.get_cluster_distance(latent_features)
        cluster_labels = clustering.predict(latent_features)
        logger.info(f"✓ Clustering updated - Distance mean: {np.mean(cluster_distance):.6f}")
        
        logger.info("\n[STEP 2e] Incrementally updating Anomaly Detector...")
        anomaly.update(X_scaled)
        anomaly_score = anomaly.predict(X_scaled)
        logger.info(f"✓ Anomaly detector updated - Score mean: {np.mean(anomaly_score):.6f}")
        
        training_type = "incremental"
    
    # --- Save all models ---
    logger.info("\n[STEP 3] Saving all model artifacts...")
    try:
        save_all_models(autoencoder, clustering, anomaly, preprocessor)
    except Exception as e:
        logger.error(f"Failed to save models: {e}")
        return
    
    # --- Fuse scores ---
    logger.info("\n[STEP 4] Fusing component scores...")
    final_risk_score = fuse_scores(cluster_distance, anomaly_score, reconstruction_error)
    
    # --- Categorize risk ---
    logger.info("\n[STEP 5] Categorizing risk levels...")
    risk_category = categorize_risk(final_risk_score)
    
    # --- Database operations ---
    logger.info("\n[STEP 6] Storing results to database...")
    conn = get_db_connection_wrapper()
    
    if conn is None:
        logger.error("Failed to connect to database")
        return
    
    try:
        # Register training event
        model_name = "Churn_Autoencoder_KMeans_IsolationForest"
        model_bundle_id = insert_model_registry(conn, model_name, training_type, len(X))
        
        if model_bundle_id is None:
            logger.error("Failed to register training event")
            return
        
        # Evaluation metrics
        logger.info("\n[STEP 7] Evaluating models...")
        metrics = evaluate_all(
            reconstruction_error,
            latent_features,
            cluster_labels,
            anomaly.model,
            X_scaled
        )
        insert_model_evaluation(conn, model_bundle_id, metrics)
        
        # Training summary
        logger.info("\n[STEP 8] Storing training summary...")
        total_customers = len(final_risk_score)
        stable = np.sum(risk_category == "Stable")
        at_risk = np.sum(risk_category == "At Risk")
        high_risk = np.sum(risk_category == "High Risk")
        
        insert_training_summary(conn, model_bundle_id, total_customers, stable, at_risk, high_risk)
        
        # Customer risk scores
        logger.info("\n[STEP 9] Storing customer risk scores...")
        all_customers = []
        for i in range(total_customers):
            row = (
                int(customer_ids[i]),
                int(model_bundle_id),
                float(cluster_distance[i]),
                float(anomaly_score[i]),
                float(reconstruction_error[i]),
                float(final_risk_score[i]),
                str(risk_category[i])
            )
            all_customers.append(row)
        
        insert_customer_scores(conn, all_customers)
        
        # Summary Report
        duration = time.time() - start_time
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ {training_type.upper()} TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"\nTraining Details:")
        logger.info(f"  Training Type: {training_type}")
        logger.info(f"  Model Bundle ID: {model_bundle_id}")
        logger.info(f"  Timestamp: {datetime.now()}")
        logger.info(f"  Customers Processed: {total_customers}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"\nCustomer Distribution:")
        logger.info(f"  Stable: {stable} ({100*stable/total_customers:.1f}%)")
        logger.info(f"  At Risk: {at_risk} ({100*at_risk/total_customers:.1f}%)")
        logger.info(f"  High Risk: {high_risk} ({100*high_risk/total_customers:.1f}%)")
        logger.info(f"\nModel Metrics:")
        logger.info(f"  Reconstruction Error Mean: {metrics['reconstruction_error_mean']:.6f}")
        logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"  Anomaly Contamination Rate: {metrics['anomaly_contamination_rate']:.4f}")
        logger.info("="*70)
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if conn:
            conn.close()
            logger.info("✓ Database connection closed")


if __name__ == "__main__":
    main()