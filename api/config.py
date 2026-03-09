from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "mysql+pymysql://root:root@localhost:3306/customer_churn_system"
    
    # API
    api_title: str = "Customer Churn Risk Detection API"
    api_version: str = "1.0.0"
    api_description: str = "ML pipeline API for detecting customer churn risk"
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Model paths
    autoencoder_model_path: str = "trained_model/autoencoder_model.pkl"
    clustering_model_path: str = "trained_model/clustering_model.pkl"
    anomaly_model_path: str = "trained_model/anomaly_model.pkl"
    preprocessor_path: str = "trained_model/preprocessor.pkl"
    
    # Risk thresholds
    stable_threshold: float = 0.33
    at_risk_threshold: float = 0.67
    high_risk_threshold: float = 1.0
    
    class Config:
        env_file = ".env"

settings = Settings()