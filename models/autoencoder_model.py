# models/autoencoder_model.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoencoderModel:
    """
    Wrapper class for Autoencoder anomaly detection.
    Compatible with your existing implementation while providing class interface for FastAPI.
    """
    
    def __init__(self, input_dim=10, encoding_dim=6, hidden_dim=8, epochs=50):
        """
        Initialize the autoencoder model.
        
        Args:
            input_dim (int): Input feature dimension
            encoding_dim (int): Bottleneck dimension
            hidden_dim (int): Hidden layer dimension
            epochs (int): Training epochs
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.reconstruction_error = None
        self.history = None
    
    def build_model(self):
        """Build the autoencoder architecture"""
        logger.info(f"Building Autoencoder: input_dim={self.input_dim}, encoding_dim={self.encoding_dim}, hidden_dim={self.hidden_dim}")
        
        # === Encoder ===
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.hidden_dim, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='latent')(encoded)
        
        # === Decoder ===
        decoded = layers.Dense(self.hidden_dim, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # === Models ===
        self.autoencoder = keras.Model(input_layer, decoded, name="autoencoder")
        self.encoder = keras.Model(input_layer, encoded, name="encoder")
        
        # Compile
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("✓ Autoencoder architecture built")
    
    def train(self, X_scaled, validation_split=0.1, batch_size=32, verbose=1):
        """
        Train autoencoder on scaled features.
        
        Args:
            X_scaled (np.ndarray): Scaled feature matrix
            validation_split (float): Validation split ratio
            batch_size (int): Batch size
            verbose (int): Verbosity level
            
        Returns:
            dict: Training results
        """
        logger.info("="*70)
        logger.info("TRAINING AUTOENCODER")
        logger.info("="*70)
        logger.info(f"Data shape: {X_scaled.shape}")
        
        # Build model if not already built
        if self.autoencoder is None:
            self.build_model()
        
        # Scale data
        X_train = self.scaler.fit_transform(X_scaled)
        
        logger.info(f"Starting training: {self.epochs} epochs, batch_size={batch_size}")
        self.history = self.autoencoder.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=True
        )
        
        # Compute reconstruction error
        logger.info("Computing reconstruction errors...")
        X_pred = self.autoencoder.predict(X_train, verbose=0)
        self.reconstruction_error = np.mean(np.square(X_train - X_pred), axis=1)
        
        self.is_trained = True
        
        logger.info(f"✓ Autoencoder training complete")
        logger.info(f"  Mean reconstruction error: {np.mean(self.reconstruction_error):.6f}")
        logger.info(f"  Min: {np.min(self.reconstruction_error):.6f}, Max: {np.max(self.reconstruction_error):.6f}")
        
        return {
            'autoencoder': self.autoencoder,
            'encoder': self.encoder,
            'reconstruction_error': self.reconstruction_error,
            'history': self.history
        }
    
    def get_reconstruction_error(self, X):
        """
        Get reconstruction error for input data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Normalized reconstruction errors [0, 1]
        """
        if not self.is_trained or self.autoencoder is None:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_pred = self.autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)
        
        # Normalize
        if len(mse) > 0:
            min_val = mse.min()
            max_val = mse.max()
            if max_val > min_val:
                errors = (mse - min_val) / (max_val - min_val)
            else:
                errors = np.zeros_like(mse)
        else:
            errors = np.array([0.0])
        
        return errors
    
    def predict(self, X):
        """Alias for get_reconstruction_error"""
        return self.get_reconstruction_error(X)
    
    def get_encoding(self, X):
        """
        Get the encoded representation of the input.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Encoded representations
        """
        if not self.is_trained or self.encoder is None:
            raise ValueError("Model must be trained before encoding")
        
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)
    
    def save(self, filepath):
        """Save the trained model to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save metadata
            metadata_path = filepath.replace('.pkl', '_meta.pkl')
            joblib.dump({
                'input_dim': self.input_dim,
                'encoding_dim': self.encoding_dim,
                'hidden_dim': self.hidden_dim,
                'epochs': self.epochs,
                'is_trained': self.is_trained,
                'scaler': self.scaler
            }, metadata_path)
            
            # Save models
            if self.autoencoder is not None:
                model_path = filepath.replace('.pkl', '_model.h5')
                self.autoencoder.save(model_path)
            
            if self.encoder is not None:
                encoder_path = filepath.replace('.pkl', '_encoder.h5')
                self.encoder.save(encoder_path)
            
            logger.info(f"✓ Autoencoder saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Error saving autoencoder: {e}")
            raise
    
    @staticmethod
    def load(filepath):
        """Load a trained model from disk"""
        try:
            # Load metadata
            metadata_path = filepath.replace('.pkl', '_meta.pkl')
            meta = joblib.load(metadata_path)
            
            # Create instance
            instance = AutoencoderModel(
                input_dim=meta['input_dim'],
                encoding_dim=meta['encoding_dim'],
                hidden_dim=meta['hidden_dim'],
                epochs=meta['epochs']
            )
            
            # Load models
            try:
                model_path = filepath.replace('.pkl', '_model.h5')
                instance.autoencoder = keras.models.load_model(model_path)
            except:
                logger.warning("Could not load autoencoder model")
            
            try:
                encoder_path = filepath.replace('.pkl', '_encoder.h5')
                instance.encoder = keras.models.load_model(encoder_path)
            except:
                logger.warning("Could not load encoder model")
            
            instance.scaler = meta['scaler']
            instance.is_trained = meta['is_trained']
            
            logger.info(f"✓ Autoencoder loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"✗ Error loading autoencoder: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the trained model"""
        return {
            "model_type": "Autoencoder",
            "is_trained": self.is_trained,
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dim": self.hidden_dim
        }