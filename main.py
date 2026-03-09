from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
import os
import numpy as np

# Add paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from database.db_connection import get_db_connection, close_connection
from models.anomaly_model import AnomalyDetector
from models.clustering_model import ClusteringModel
from models.autoencoder_model import AutoencoderModel
from scripts.preprocess_data import DataPreprocessor
from api.routes import dashboard, customers, inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app state
app_state = {
    'models_loaded': False,
    'autoencoder': None,
    'clustering_model': None,
    'anomaly_detector': None,
    'preprocessor': None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting application...")
    logger.info("Attempting to load ML models...")
    
    try:
        load_models()
        app_state['models_loaded'] = True
        logger.info("✓ ML models loaded successfully")
    except Exception as e:
        logger.warning(f"⚠ Could not load pre-trained models: {e}")
        logger.info("Initializing with fresh models...")
        try:
            initialize_fresh_models()
            logger.info("✓ Fresh models initialized")
        except Exception as e2:
            logger.error(f"✗ Failed to initialize models: {e2}")
            app_state['models_loaded'] = False
    
    yield
    
    logger.info("Shutting down application...")
    cleanup_models()

def initialize_fresh_models():
    """Initialize fresh models without pre-training"""
    logger.info("Initializing fresh ML models...")
    
    # Create untrained models
    app_state['preprocessor'] = DataPreprocessor()
    app_state['autoencoder'] = AutoencoderModel(input_dim=10, encoding_dim=6)
    app_state['clustering_model'] = ClusteringModel(n_clusters=5)
    app_state['anomaly_detector'] = AnomalyDetector(contamination='auto')
    
    logger.info("✓ Fresh models initialized (not yet trained)")

def load_models():
    """Load all ML models from disk"""
    try:
        logger.info("Loading preprocessor...")
        app_state['preprocessor'] = DataPreprocessor.load('trained_model/preprocessor.pkl')
        
        logger.info("Loading autoencoder...")
        app_state['autoencoder'] = AutoencoderModel.load('trained_model/autoencoder_model.pkl')
        
        logger.info("Loading clustering model...")
        app_state['clustering_model'] = ClusteringModel.load('trained_model/clustering_model.pkl')
        
        logger.info("Loading anomaly detector...")
        app_state['anomaly_detector'] = AnomalyDetector.load('trained_model/anomaly_model.pkl')
        
        logger.info("✓ All pre-trained models loaded successfully")
        app_state['models_loaded'] = True
    except FileNotFoundError as e:
        logger.warning(f"Pre-trained models not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def cleanup_models():
    """Cleanup before shutdown"""
    for key in list(app_state.keys()):
        if key != 'models_loaded':
            app_state[key] = None
    logger.info("Cleanup complete")

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Risk Detection API",
    description="ML pipeline API for detecting customer churn risk using unsupervised learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.models = app_state

# Include routers
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(customers.router, prefix="/api/customers", tags=["Customers"])
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": app_state['models_loaded'],
        "autoencoder": app_state['autoencoder'] is not None,
        "clustering_model": app_state['clustering_model'] is not None,
        "anomaly_detector": app_state['anomaly_detector'] is not None,
        "preprocessor": app_state['preprocessor'] is not None
    }

@app.get("/api/models/info", tags=["Models"])
async def get_model_info():
    """Get active model information"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT model_bundle_id, model_name, training_type, training_date, training_data_size, is_active
            FROM model_registry
            WHERE is_active = TRUE
            ORDER BY training_date DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        if result:
            return {
                "model_bundle_id": result[0],
                "model_name": result[1],
                "training_type": result[2],
                "training_date": result[3].isoformat() if result[3] else None,
                "training_data_size": result[4],
                "is_active": bool(result[5])
            }
        else:
            return {
                "model_bundle_id": None,
                "model_name": "Unsupervised Combined Model v1.0",
                "training_type": "initial",
                "training_date": None,
                "training_data_size": 0,
                "is_active": False
            }
    except Exception as e:
        logger.warning(f"Error fetching model info: {e}")
        return {
            "model_bundle_id": None,
            "model_name": "Unsupervised Combined Model v1.0",
            "training_type": "initial",
            "training_date": None,
            "training_data_size": 0,
            "is_active": False
        }
    finally:
        close_connection(connection)

@app.get("/api/models/metrics", tags=["Models"])
async def get_model_metrics():
    """Get evaluation metrics of active model"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT me.reconstruction_error_mean, me.silhouette_score, me.anomaly_contamination_rate
            FROM model_evaluation me
            JOIN model_registry mr ON me.model_bundle_id = mr.model_bundle_id
            WHERE mr.is_active = TRUE
            ORDER BY me.created_at DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        if result:
            return {
                "reconstruction_error_mean": float(result[0]) if result[0] else 0.0,
                "silhouette_score": float(result[1]) if result[1] else 0.0,
                "anomaly_contamination_rate": float(result[2]) if result[2] else 0.0
            }
        else:
            return {
                "reconstruction_error_mean": 0.0,
                "silhouette_score": 0.0,
                "anomaly_contamination_rate": 0.0
            }
    except Exception as e:
        logger.warning(f"Error fetching metrics: {e}")
        return {
            "reconstruction_error_mean": 0.0,
            "silhouette_score": 0.0,
            "anomaly_contamination_rate": 0.0
        }
    finally:
        close_connection(connection)

@app.get("/api/models/feature-importance", tags=["Models"])
async def get_feature_importance():
    """Get feature importance for risk scoring"""
    return [
        {"feature": "Monthly Charge", "importance": 0.25},
        {"feature": "Data Usage", "importance": 0.20},
        {"feature": "Customer Service Calls", "importance": 0.18},
        {"feature": "Account Weeks", "importance": 0.15},
        {"feature": "Day Minutes", "importance": 0.12},
        {"feature": "Overage Fee", "importance": 0.10}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )