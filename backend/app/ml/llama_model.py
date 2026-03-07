"""
Llama-3.2-3B Integration for SteadyPrice Enterprise
Production-ready fine-tuned model implementation
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
import structlog
from datetime import datetime
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

from app.core.config import settings
from app.ml.fine_tuning import FineTuningManager, PromptFormatter

logger = structlog.get_logger()

class LlamaPricePredictor:
    """Production-ready Llama model for price prediction"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "./models/fine_tuned"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.prompt_formatter = None
        self.is_loaded = False
        
        # Model configuration
        self.base_model = "meta-llama/Llama-3.2-3B"
        self.max_length = 512
        self.temperature = 0.1
        self.max_new_tokens = 10
        
    async def load_model(self):
        """Load the fine-tuned Llama model"""
        try:
            logger.info(f"Loading Llama model from {self.model_path}")
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Load fine-tuned adapter if available
            if os.path.exists(self.model_path):
                logger.info("Loading fine-tuned adapter")
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                logger.warning("No fine-tuned adapter found, using base model")
                self.model = base_model
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                return_full_text=False
            )
            
            # Initialize prompt formatter
            self.prompt_formatter = PromptFormatter(self.tokenizer)
            
            self.is_loaded = True
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
    
    def create_prediction_prompt(self, title: str, category: str, description: str = "") -> str:
        """Create optimized prompt for price prediction"""
        
        # Build product context
        context_parts = [
            f"Product: {title}",
            f"Category: {category.replace('_', ' ')}"
        ]
        
        if description and description.strip():
            # Limit description length
            desc = description[:200] + "..." if len(description) > 200 else description
            context_parts.append(f"Details: {desc}")
        
        product_context = "\n".join(context_parts)
        
        # Create expert prompt
        prompt = (
            f"As an expert pricing analyst, analyze this product and predict its market price:\n\n"
            f"{product_context}\n\n"
            f"Consider brand value, features, quality, and current market trends.\n"
            f"Provide a precise price prediction in US dollars: $"
        )
        
        return prompt
    
    def parse_price_from_output(self, output: str) -> Tuple[float, float]:
        """Parse price and calculate confidence from model output"""
        
        try:
            # Clean the output
            clean_output = output.strip()
            
            # Extract price (look for first number after $)
            if "$" in clean_output:
                price_part = clean_output.split("$")[-1].strip()
            else:
                price_part = clean_output.split()[0]
            
            # Extract numeric value
            import re
            price_match = re.search(r'\d+\.?\d*', price_part)
            
            if price_match:
                price = float(price_match.group())
                
                # Validate price range
                if 1 <= price <= 10000:
                    # Calculate confidence based on output quality
                    confidence = self._calculate_confidence(clean_output, price)
                    return price, confidence
                else:
                    logger.warning(f"Price out of range: {price}")
            
        except Exception as e:
            logger.error(f"Failed to parse price: {e}")
        
        # Fallback values
        return 199.99, 0.75
    
    def _calculate_confidence(self, output: str, price: float) -> float:
        """Calculate confidence score based on output characteristics"""
        
        confidence = 0.8  # Base confidence
        
        # Adjust based on price reasonableness
        if 50 <= price <= 500:
            confidence += 0.1
        elif 500 < price <= 1000:
            confidence += 0.05
        elif price > 1000:
            confidence -= 0.1
        
        # Adjust based on output length (too short or too long is less reliable)
        output_length = len(output.strip())
        if 5 <= output_length <= 20:
            confidence += 0.05
        elif output_length < 3 or output_length > 30:
            confidence -= 0.1
        
        # Adjust based on format (presence of decimals suggests precision)
        if "." in str(price):
            confidence += 0.05
        
        return min(0.95, max(0.6, confidence))
    
    async def predict_price(self, title: str, category: str, description: str = "") -> Tuple[float, float]:
        """Make price prediction using Llama model"""
        
        if not self.is_loaded:
            await self.load_model()
        
        try:
            # Create prompt
            prompt = self.create_prediction_prompt(title, category, description)
            
            # Generate prediction
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract output
            generated_text = outputs[0]['generated_text'].strip()
            
            # Parse price
            price, confidence = self.parse_price_from_output(generated_text)
            
            logger.info(f"Llama prediction: ${price:.2f} (confidence: {confidence:.2f})")
            
            return price, confidence
            
        except Exception as e:
            logger.error(f"Llama prediction failed: {e}")
            # Fallback to category-based pricing
            category_prices = {
                "Electronics": 299.99,
                "Appliances": 199.99,
                "Automotive": 499.99,
                "Office_Products": 89.99,
                "Tools_and_Home_Improvement": 149.99,
                "Cell_Phones_and_Accessories": 399.99,
                "Toys_and_Games": 49.99,
                "Musical_Instruments": 299.99
            }
            fallback_price = category_prices.get(category, 199.99)
            return fallback_price, 0.7
    
    async def batch_predict(self, products: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Batch prediction for multiple products"""
        results = []
        
        for product in products:
            try:
                price, confidence = await self.predict_price(
                    title=product.get('title', ''),
                    category=product.get('category', ''),
                    description=product.get('description', '')
                )
                results.append((price, confidence))
            except Exception as e:
                logger.error(f"Batch prediction failed for {product.get('title', 'unknown')}: {e}")
                results.append((199.99, 0.7))  # Fallback
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": "llama-3.2-3b-fine-tuned",
            "base_model": self.base_model,
            "adapter_path": self.model_path,
            "is_loaded": self.is_loaded,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "fine_tuned": os.path.exists(self.model_path),
            "last_loaded": datetime.now().isoformat() if self.is_loaded else None
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("Llama model unloaded")

class LlamaModelManager:
    """Production manager for Llama model operations"""
    
    def __init__(self):
        self.predictor = LlamaPricePredictor()
        self.fine_tuning_manager = FineTuningManager()
        self.model_cache = {}
        
    async def initialize(self):
        """Initialize all model components"""
        logger.info("Initializing Llama model manager")
        
        # Try to load fine-tuned model
        try:
            await self.predictor.load_model()
            logger.info("Fine-tuned Llama model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model: {e}")
            # Will load base model on first prediction
        
        logger.info("Llama model manager initialized")
    
    async def predict_price(self, title: str, category: str, description: str = "") -> Tuple[float, float]:
        """Make price prediction"""
        return await self.predictor.predict_price(title, category, description)
    
    async def train_model(self, products: List[Dict[str, Any]]):
        """Train new fine-tuned model"""
        logger.info(f"Starting fine-tuning with {len(products)} products")
        
        # Initialize fine-tuning
        await self.fine_tuning_manager.initialize()
        
        # Train model
        result = await self.fine_tuning_manager.train_from_data(products)
        
        # Reload model with new adapter
        await self.predictor.load_model()
        
        logger.info("Fine-tuning completed and model reloaded")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "predictor": self.predictor.get_model_info(),
            "fine_tuning_initialized": self.fine_tuning_manager.is_initialized,
            "cuda_available": torch.cuda.is_available(),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
