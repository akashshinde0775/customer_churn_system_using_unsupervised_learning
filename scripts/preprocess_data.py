import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, 'data', 'fetched', 'customer_features_fetched.csv')
PROCESSED_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'customer_features_processed.csv')
SCALER_PARAMS_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'scaler_params.npy')

# Columns to drop
DROP_COLS = ['customer_id', 'created_at', 'churn', 'Churn', 'CHURN']  
REMOVE_OUTLIERS = False  # Set to True to remove outliers, False to keep all data


# ========== Step 1: Data Validation ==========
def validate_data(df):
    """Validate and inspect the loaded dataset"""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: DATA VALIDATION")
    logger.info("="*70)
    
    logger.info(f"\n✓ Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"\n✓ Column Names:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"   {i}. {col} ({df[col].dtype})")
    
    logger.info(f"\n✓ Data Types Summary:")
    logger.info(f"{df.dtypes}")
    
    logger.info(f"\n✓ First 5 Rows:")
    logger.info(f"{df.head()}")


# ========== Step 2: Missing Value Handling ==========
def handle_missing_values(df):
    """Handle missing values by filling with mean"""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: MISSING VALUE HANDLING")
    logger.info("="*70)
    
    missing_count = df.isnull().sum()
    if missing_count.sum() == 0:
        logger.info("\n✓ No missing values found!")
        return df
    
    logger.info(f"\n✓ Missing values before handling:")
    logger.info(f"{missing_count[missing_count > 0]}")
    
    # Fill numeric columns with mean
    df_filled = df.fillna(df.mean(numeric_only=True))
    
    # Fill remaining non-numeric with forward fill/back fill
    df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"\n✓ Missing values after handling:")
    logger.info(f"{df_filled.isnull().sum().sum()} total missing values")
    
    return df_filled


# ========== Step 3: Feature Type Conversion ==========
def convert_feature_types(df):
    """Convert all features to numeric type"""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: FEATURE TYPE CONVERSION")
    logger.info("="*70)
    
    logger.info(f"\n✓ Converting features to numeric...")
    
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            logger.warning(f"   Warning: Could not convert {col}: {e}")
    
    logger.info(f"✓ All features converted to numeric type")
    
    return df


# ========== Step 4: Outlier Handling ==========
def handle_outliers(df, remove_outliers=False):
    """Optionally remove outliers using IQR method"""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: OUTLIER HANDLING")
    logger.info("="*70)
    
    if not remove_outliers:
        logger.info(f"\n✓ Outlier removal DISABLED")
        logger.info(f"✓ Keeping all rows: {len(df)}")
        return df
    
    # If outlier removal is enabled
    OUTLIER_FACTOR = 1.5
    logger.info(f"\n✓ Using IQR factor: {OUTLIER_FACTOR}")
    logger.info(f"✓ Initial rows: {len(df)}")
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - OUTLIER_FACTOR * IQR
    upper_bound = Q3 + OUTLIER_FACTOR * IQR
    
    outlier_mask = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
    
    df_clean = df[~outlier_mask].copy()
    
    rows_removed = len(df) - len(df_clean)
    logger.info(f"✓ Rows removed as outliers: {rows_removed}")
    logger.info(f"✓ Remaining rows: {len(df_clean)}")
    
    return df_clean


# ========== Step 5: Feature Selection ==========
def select_features(df):
    """Select relevant features (remove ID, timestamps, labels)"""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: FEATURE SELECTION")
    logger.info("="*70)
    
    cols_to_drop = [col for col in DROP_COLS if col in df.columns]
    
    if cols_to_drop:
        logger.info(f"\n✓ Dropping columns: {cols_to_drop}")
        df_selected = df.drop(columns=cols_to_drop, axis=1)
    else:
        logger.info(f"\n✓ No columns to drop")
        df_selected = df.copy()
    
    logger.info(f"\n✓ Selected {len(df_selected.columns)} features:")
    for i, col in enumerate(df_selected.columns, 1):
        logger.info(f"   {i}. {col}")
    
    logger.info(f"\n✓ Feature matrix shape: {df_selected.shape}")
    
    return df_selected


# ========== Step 6: Feature Scaling ==========
def scale_features(X):
    """Scale features using StandardScaler"""
    logger.info("\n" + "="*70)
    logger.info("STEP 6: FEATURE SCALING (StandardScaler)")
    logger.info("="*70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"\n✓ Scaling applied using StandardScaler")
    logger.info(f"✓ Scaled data shape: {X_scaled.shape}")
    logger.info(f"✓ Mean of scaled features (should be ~0): {X_scaled.mean(axis=0)[:5]}")
    logger.info(f"✓ Std of scaled features (should be ~1): {X_scaled.std(axis=0)[:5]}")
    
    return X_scaled, scaler


# ========== Main Preprocessing Pipeline ==========
def preprocess_pipeline(input_file=INPUT_FILE):
    """Execute the complete preprocessing pipeline"""
    logger.info("\n" + "="*70)
    logger.info("CUSTOMER CHURN SYSTEM - DATA PREPROCESSING PIPELINE")
    logger.info("="*70)
    
    # Step 1: Load and validate
    logger.info(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    validate_data(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Feature type conversion
    df = convert_feature_types(df)
    
    # Step 4: Outlier handling
    df = handle_outliers(df, remove_outliers=REMOVE_OUTLIERS)
    
    # Step 5: Feature selection
    df_selected = select_features(df)
    
    # Step 6: Feature scaling
    X_scaled, scaler = scale_features(df_selected)
    
    # Save processed data
    logger.info("\n" + "="*70)
    logger.info("SAVING PROCESSED DATA")
    logger.info("="*70)
    
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    
    processed_df = pd.DataFrame(X_scaled, columns=df_selected.columns)
    processed_df.to_csv(PROCESSED_FILE, index=False)
    logger.info(f"\n✓ Processed data saved: {PROCESSED_FILE}")
    logger.info(f"   Shape: {processed_df.shape}")
    
    # Save scaler parameters
    scaler_params = {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    }
    np.save(SCALER_PARAMS_FILE, scaler_params, allow_pickle=True)
    logger.info(f"✓ Scaler parameters saved: {SCALER_PARAMS_FILE}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PREPROCESSING COMPLETE ✓")
    logger.info("="*70)
    logger.info(f"\nFinal Feature Matrix (X_scaled):")
    logger.info(f"   Shape: {X_scaled.shape}")
    logger.info(f"   Rows (samples): {X_scaled.shape[0]}")
    logger.info(f"   Columns (features): {X_scaled.shape[1]}")
    logger.info(f"   Features: {list(df_selected.columns)}")
    
    return X_scaled, df_selected.columns, scaler


# ========== DataPreprocessor Class ==========
class DataPreprocessor:
    """
    Wrapper class for data preprocessing.
    Compatible with your existing preprocessing functions while providing class interface for FastAPI.
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage',
            'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee', 'RoamMins'
        ]
    
    def fit(self, X):
        """
        Fit the preprocessor on training data.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        try:
            logger.info(f"Fitting preprocessor on {X.shape[0]} samples with {X.shape[1]} features...")
            self.scaler.fit(X)
            self.is_fitted = True
            logger.info("✓ Preprocessor fitted successfully")
            return self
        except Exception as e:
            logger.error(f"✗ Error fitting preprocessor: {e}")
            raise
    
    def transform(self, X):
        """
        Transform data using fitted preprocessor.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        try:
            return self.scaler.transform(X)
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            raise
    
    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Transform scaled data back to original scale.
        
        Args:
            X_scaled (np.ndarray): Scaled feature matrix
            
        Returns:
            np.ndarray: Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        try:
            return self.scaler.inverse_transform(X_scaled)
        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            raise
    
    def save(self, filepath):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump({
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names
            }, filepath)
            logger.info(f"✓ Preprocessor saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Error saving preprocessor: {e}")
            raise
    
    @staticmethod
    def load(filepath):
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath (str): Path to load the preprocessor from
            
        Returns:
            DataPreprocessor: Loaded DataPreprocessor instance
        """
        try:
            data = joblib.load(filepath)
            
            instance = DataPreprocessor()
            instance.scaler = data['scaler']
            instance.is_fitted = data['is_fitted']
            instance.feature_names = data['feature_names']
            
            logger.info(f"✓ Preprocessor loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Error loading preprocessor: {e}")
            raise
    
    def get_feature_stats(self):
        """Get statistics about fitted features"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return {
            "feature_names": self.feature_names,
            "means": self.scaler.mean_.tolist(),
            "stds": self.scaler.scale_.tolist(),
            "n_features": len(self.feature_names)
        }


if __name__ == "__main__":
    X_scaled, feature_names, scaler = preprocess_pipeline()