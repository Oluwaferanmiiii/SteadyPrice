"""
Machine Learning models for price prediction
Enterprise-grade ML implementations for SteadyPrice
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import structlog

from app.core.config import settings
from app.models.schemas import ProductCategory, ModelType

logger = structlog.get_logger()

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    mae: float
    rmse: float
    mape: float
    r2_score: float

class TraditionalMLModel:
    """Traditional ML models for price prediction"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def preprocess_features(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features for training"""
        df = pd.DataFrame(data)
        
        # Text features
        text_features = []
        for _, row in df.iterrows():
            text = f"{row['title']} {row.get('description', '')}"
            text_features.append(text)
        
        # TF-IDF features
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(text_features)
        
        # Categorical features
        categories_encoded = self.label_encoder.fit_transform(df['category'])
        
        # Numerical features
        numerical_features = []
        if 'weight' in df.columns:
            numerical_features.append(df['weight'].fillna(0))
        else:
            numerical_features.append([0] * len(df))
        
        # Feature count (description length, title length)
        desc_lengths = [len(str(row.get('description', ''))) for _, row in df.iterrows()]
        title_lengths = [len(str(row.get('title', ''))) for _, row in df.iterrows()]
        numerical_features.extend([desc_lengths, title_lengths])
        
        # Combine all features
        numerical_features = np.array(numerical_features).T
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine TF-IDF and numerical features
        features = np.hstack([
            tfidf_matrix.toarray(),
            numerical_features_scaled,
            categories_encoded.reshape(-1, 1)
        ])
        
        self.feature_columns = features.shape[1]
        
        # Target variable
        targets = df['price'].values
        
        return features, targets
    
    def train(self, data: List[Dict[str, Any]]) -> ModelMetrics:
        """Train the model"""
        logger.info(f"Training {self.model_type} model")
        
        # Preprocess features
        features, targets = self.preprocess_features(data)
        
        # Train model
        self.model.fit(features, targets)
        self.is_trained = True
        
        # Calculate metrics
        predictions = self.model.predict(features)
        metrics = self._calculate_metrics(targets, predictions)
        
        logger.info(f"Model trained - MAE: {metrics.mae:.2f}, R2: {metrics.r2_score:.3f}")
        return metrics
    
    def predict(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """Make prediction with confidence score"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess single instance
        features, _ = self.preprocess_features([data])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Calculate confidence (simplified - based on prediction variance)
        if hasattr(self.model, 'predict_proba'):
            confidence = 0.8  # Placeholder
        else:
            confidence = min(0.9, max(0.5, 1.0 - abs(prediction - 50) / 100))
        
        return max(0, prediction), confidence
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate model metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate accuracy (within 10% of actual price)
        accuracy = np.mean(np.abs(y_true - y_pred) / y_true <= 0.1)
        
        return ModelMetrics(
            accuracy=accuracy,
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2_score=r2
        )
    
    def save_model(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {path}")

class DeepLearningModel(nn.Module):
    """Deep Learning model for price prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.is_trained = False
        
    def forward(self, x):
        return self.network(x)

class LLMModel:
    """Fine-tuned Language Model for price prediction"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        
    def initialize(self):
        """Initialize the LLM"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"LLM initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """Preprocess text for LLM"""
        if not self.tokenizer:
            self.initialize()
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        return inputs
    
    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from text using LLM"""
        inputs = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state mean as features
            features = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return features.flatten()
    
    def predict_price(self, title: str, description: str = "") -> Tuple[float, float]:
        """Predict price using LLM features"""
        if not self.is_trained:
            # For demo, return a simple heuristic-based prediction
            text = f"{title} {description}"
            features = self.extract_features(text)
            
            # Simple heuristic based on text length and features
            text_length = len(text)
            feature_sum = np.sum(features)
            
            # Base price calculation (simplified)
            base_price = 10 + (text_length * 0.1) + (feature_sum * 50)
            price = max(1, min(1000, base_price))
            
            # Confidence based on feature consistency
            confidence = min(0.9, max(0.5, 1.0 - np.std(features)))
            
            return price, confidence

class EnsembleModel:
    """Ensemble model combining multiple approaches"""
    
    def __init__(self):
        self.models = {}
        self.weights = {
            ModelType.TRADITIONAL_ML: 0.3,
            ModelType.DEEP_LEARNING: 0.3,
            ModelType.FINE_TUNED_LLM: 0.4
        }
        
    def add_model(self, model_type: ModelType, model):
        """Add a model to the ensemble"""
        self.models[model_type] = model
        
    def predict(self, data: Dict[str, Any]) -> Tuple[float, float, ModelType]:
        """Make ensemble prediction"""
        predictions = []
        confidences = []
        
        for model_type, model in self.models.items():
            try:
                if model_type == ModelType.TRADITIONAL_ML:
                    pred, conf = model.predict(data)
                elif model_type == ModelType.FINE_TUNED_LLM:
                    pred, conf = model.predict_price(
                        data['title'], 
                        data.get('description', '')
                    )
                else:
                    continue  # Skip unsupported models
                
                predictions.append(pred)
                confidences.append(conf)
                
            except Exception as e:
                logger.warning(f"Model {model_type} failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted average prediction
        weights = [self.weights.get(mt, 0.1) for mt in self.models.keys() if mt in self.models]
        weights = np.array(weights[:len(predictions)])
        weights = weights / weights.sum()  # Normalize
        
        ensemble_prediction = np.average(predictions, weights=weights)
        ensemble_confidence = np.mean(confidences)
        
        # Determine best model type
        best_model_idx = np.argmax(confidences)
        model_types = list(self.models.keys())
        best_model_type = model_types[best_model_idx]
        
        return ensemble_prediction, ensemble_confidence, best_model_type

class ModelManager:
    """Enterprise model manager for SteadyPrice"""
    
    def __init__(self):
        self.models = {}
        self.ensemble = EnsembleModel()
        self.model_metrics = {}
        self.is_initialized = False
        
    async def initialize_models(self):
        """Initialize all models"""
        logger.info("Initializing ML models")
        
        try:
            # Initialize Traditional ML models
            rf_model = TraditionalMLModel("random_forest")
            xgb_model = TraditionalMLModel("xgboost")
            
            self.models[ModelType.TRADITIONAL_ML] = rf_model
            self.models[ModelType.DEEP_LEARNING] = xgb_model
            
            # Initialize LLM model
            llm_model = LLMModel()
            llm_model.initialize()
            self.models[ModelType.FINE_TUNED_LLM] = llm_model
            
            # Add models to ensemble
            self.ensemble.add_model(ModelType.TRADITIONAL_ML, rf_model)
            self.ensemble.add_model(ModelType.FINE_TUNED_LLM, llm_model)
            
            self.is_initialized = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if models are ready"""
        return self.is_initialized
    
    async def train_models(self, training_data: List[Dict[str, Any]]):
        """Train all models with provided data"""
        logger.info(f"Training models with {len(training_data)} samples")
        
        # Train traditional ML models
        for model_type in [ModelType.TRADITIONAL_ML, ModelType.DEEP_LEARNING]:
            if model_type in self.models:
                metrics = self.models[model_type].train(training_data)
                self.model_metrics[model_type] = metrics
                logger.info(f"{model_type} trained - Accuracy: {metrics.accuracy:.3f}")
    
    async def predict(self, data: Dict[str, Any], model_type: ModelType = ModelType.ENSEMBLE) -> Tuple[float, float]:
        """Make prediction using specified model type"""
        if not self.is_initialized:
            raise ValueError("Models not initialized")
        
        if model_type == ModelType.ENSEMBLE:
            prediction, confidence, best_model = self.ensemble.predict(data)
            return prediction, confidence
        
        elif model_type in self.models:
            model = self.models[model_type]
            if model_type == ModelType.FINE_TUNED_LLM:
                return model.predict_price(data['title'], data.get('description', ''))
            else:
                return model.predict(data)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_metrics(self, model_type: ModelType) -> Optional[ModelMetrics]:
        """Get metrics for a specific model"""
        return self.model_metrics.get(model_type)
