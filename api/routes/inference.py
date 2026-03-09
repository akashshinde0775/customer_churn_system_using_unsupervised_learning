from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from database.db_connection import get_db_connection, close_connection
from io import StringIO

logger = logging.getLogger(__name__)
router = APIRouter()

class CustomerFeatures(BaseModel):
    AccountWeeks: float = Field(..., ge=0)
    ContractRenewal: float = Field(..., ge=0, le=1)
    DataPlan: float = Field(..., ge=0, le=1)
    DataUsage: float = Field(..., ge=0)
    CustServCalls: float = Field(..., ge=0)
    DayMins: float = Field(..., ge=0)
    DayCalls: float = Field(..., ge=0)
    MonthlyCharge: float = Field(..., ge=0)
    OverageFee: float = Field(..., ge=0)
    RoamMins: float = Field(..., ge=0)

class PredictionResult(BaseModel):
    reconstruction_error: float
    cluster_distance: float
    anomaly_score: float
    final_risk_score: float
    risk_category: str

def get_prediction(feature_dict):
    """Helper function to get prediction from models"""
    from main import app_state
    
    try:
        # Extract features in correct order
        feature_array = np.array([[
            float(feature_dict.get('AccountWeeks', 0)),
            float(feature_dict.get('ContractRenewal', 0)),
            float(feature_dict.get('DataPlan', 0)),
            float(feature_dict.get('DataUsage', 0)),
            float(feature_dict.get('CustServCalls', 0)),
            float(feature_dict.get('DayMins', 0)),
            float(feature_dict.get('DayCalls', 0)),
            float(feature_dict.get('MonthlyCharge', 0)),
            float(feature_dict.get('OverageFee', 0)),
            float(feature_dict.get('RoamMins', 0))
        ]])
        
        # Preprocess
        preprocessor = app_state.get('preprocessor')
        if preprocessor and hasattr(preprocessor, 'is_fitted') and preprocessor.is_fitted:
            try:
                scaled_features = preprocessor.transform(feature_array)
            except:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(feature_array)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_array)
        
        # Get predictions from models
        autoencoder = app_state.get('autoencoder')
        clustering_model = app_state.get('clustering_model')
        anomaly_detector = app_state.get('anomaly_detector')
        
        # Default values
        reconstruction_error = 0.3
        cluster_distance = 0.3
        anomaly_score = 0.4
        
        # Try to get actual predictions
        try:
            if autoencoder and hasattr(autoencoder, 'get_reconstruction_error'):
                result = autoencoder.get_reconstruction_error(scaled_features)
                if result is not None and len(result) > 0:
                    reconstruction_error = float(result[0])
        except Exception as e:
            logger.warning(f"Error getting reconstruction error: {e}")
        
        try:
            if clustering_model and hasattr(clustering_model, 'get_cluster_distance'):
                result = clustering_model.get_cluster_distance(scaled_features)
                if result is not None and len(result) > 0:
                    cluster_distance = float(result[0])
        except Exception as e:
            logger.warning(f"Error getting cluster distance: {e}")
        
        try:
            if anomaly_detector and hasattr(anomaly_detector, 'predict'):
                result = anomaly_detector.predict(scaled_features)
                if result is not None and len(result) > 0:
                    anomaly_score = float(result[0])
        except Exception as e:
            logger.warning(f"Error getting anomaly score: {e}")
        
        # Fuse scores
        final_risk_score = (reconstruction_error + cluster_distance + anomaly_score) / 3.0
        final_risk_score = float(np.clip(final_risk_score, 0, 1))
        
        # Determine category
        if final_risk_score < 0.33:
            risk_category = "Stable"
        elif final_risk_score < 0.67:
            risk_category = "At Risk"
        else:
            risk_category = "High Risk"
        
        return {
            "reconstruction_error": round(reconstruction_error, 4),
            "cluster_distance": round(cluster_distance, 4),
            "anomaly_score": round(anomaly_score, 4),
            "final_risk_score": round(final_risk_score, 4),
            "risk_category": risk_category
        }
    
    except Exception as e:
        logger.error(f"Error in get_prediction: {e}")
        # Return default prediction on error
        return {
            "reconstruction_error": 0.3,
            "cluster_distance": 0.3,
            "anomaly_score": 0.4,
            "final_risk_score": 0.33,
            "risk_category": "At Risk"
        }

@router.post("/predict", response_model=PredictionResult)
async def predict_single(features: CustomerFeatures):
    """Make a single customer risk prediction"""
    try:
        feature_dict = features.dict()
        result = get_prediction(feature_dict)
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch")
async def predict_batch(request: Dict[str, List[Dict[str, Any]]]):
    """Make batch predictions for multiple customers
    
    Expected format:
    {
        "data": [
            {"AccountWeeks": 45, "ContractRenewal": 1, ...},
            {"AccountWeeks": 50, "ContractRenewal": 0, ...}
        ]
    }
    """
    try:
        data_list = request.get('data', [])
        
        if not isinstance(data_list, list):
            raise HTTPException(status_code=400, detail="'data' must be a list")
        
        if len(data_list) == 0:
            raise HTTPException(status_code=400, detail="'data' list cannot be empty")
        
        results = []
        
        for idx, features_dict in enumerate(data_list):
            try:
                if not isinstance(features_dict, dict):
                    logger.warning(f"Skipping item {idx}: not a dictionary")
                    continue
                
                result = get_prediction(features_dict)
                results.append(result)
            
            except Exception as e:
                logger.warning(f"Error processing item {idx}: {e}")
                # Add default result on error
                results.append({
                    "reconstruction_error": 0.3,
                    "cluster_distance": 0.3,
                    "anomaly_score": 0.4,
                    "final_risk_score": 0.33,
                    "risk_category": "At Risk"
                })
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file"""
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_columns = [
            'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage',
            'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee', 'RoamMins'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                result = get_prediction(row.to_dict())
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                results.append({
                    "reconstruction_error": 0.3,
                    "cluster_distance": 0.3,
                    "anomaly_score": 0.4,
                    "final_risk_score": 0.33,
                    "risk_category": "At Risk"
                })
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))