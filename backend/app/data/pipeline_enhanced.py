"""
Enhanced Data Pipeline for SteadyPrice Enterprise Week 7
Amazon data processing with fine-tuning preparation
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import login
from tqdm import tqdm
import structlog
from datetime import datetime
import random

from app.core.config import settings

logger = structlog.get_logger()

@dataclass
class FineTuningSample:
    """Sample prepared for fine-tuning"""
    title: str
    description: str
    category: str
    price: float
    prompt: str
    completion: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'price': self.price,
            'prompt': self.prompt,
            'completion': self.completion
        }

class EnhancedDataPipeline:
    """Enhanced data pipeline for Week 7 fine-tuning"""
    
    def __init__(self):
        self.training_samples = []
        self.validation_samples = []
        self.test_samples = []
        self.category_stats = {}
        
        # Enhanced category mapping
        self.category_mapping = {
            "Electronics": "Electronics",
            "Appliances": "Appliances", 
            "Automotive": "Automotive",
            "Office_Products": "Office Products",
            "Tools_and_Home_Improvement": "Tools and Home Improvement",
            "Cell_Phones_and_Accessories": "Cell Phones and Accessories",
            "Toys_and_Games": "Toys and Games",
            "Musical_Instruments": "Musical Instruments"
        }
        
    async def initialize(self):
        """Initialize the enhanced pipeline"""
        logger.info("Initializing enhanced data pipeline for Week 7")
        
        # Login to HuggingFace if token is provided
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(hf_token, add_to_git_credential=True)
            logger.info("Logged into HuggingFace")
        
        # Create directories
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
        
    def create_training_prompt(self, title: str, category: str, description: str, price: float) -> Tuple[str, str]:
        """Create prompt-completion pair for fine-tuning"""
        
        # Clean and prepare inputs
        clean_title = title.strip()[:200]  # Limit length
        clean_category = self.category_mapping.get(category, category.replace('_', ' '))
        clean_description = description.strip()[:300] if description else ""
        
        # Build context
        context_parts = [
            f"Product: {clean_title}",
            f"Category: {clean_category}"
        ]
        
        if clean_description:
            context_parts.append(f"Description: {clean_description}")
        
        product_context = "\n".join(context_parts)
        
        # Create instruction prompt
        prompt = (
            f"You are an expert pricing analyst with deep knowledge of e-commerce markets. "
            f"Analyze the following product information and provide an accurate price prediction "
            f"in US dollars. Consider brand reputation, features, quality, market positioning, "
            f"and current market trends.\n\n"
            f"{product_context}\n\n"
            f"Based on your analysis, the market price is: $"
        )
        
        # Create completion
        completion = f"{price:.2f}"
        
        return prompt, completion
    
    async def load_amazon_data_for_finetuning(self, max_samples: int = 10000) -> List[FineTuningSample]:
        """Load Amazon data specifically for fine-tuning"""
        logger.info(f"Loading Amazon data for fine-tuning (max {max_samples} samples)")
        
        all_samples = []
        
        # Categories to load
        categories = [
            "raw_meta_Electronics",
            "raw_meta_Appliances",
            "raw_meta_Automotive",
            "raw_meta_Office_Products"
        ]
        
        for dataset_name in tqdm(categories, desc="Loading categories"):
            try:
                # Load dataset
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023", 
                    dataset_name, 
                    split="train",
                    trust_remote_code=True
                )
                
                category_name = dataset_name.replace("raw_meta_", "")
                samples_loaded = 0
                
                for item in tqdm(dataset, desc=f"Processing {category_name}", leave=False):
                    if samples_loaded >= max_samples // len(categories):
                        break
                    
                    try:
                        # Parse price
                        price_str = item.get('price', '')
                        if not price_str:
                            continue
                        
                        price = float(price_str.replace('$', '').replace(',', '').strip())
                        if not (1 <= price <= 1000):
                            continue
                        
                        # Get product info
                        title = item.get('title', '').strip()
                        if not title or len(title) < 10:
                            continue
                        
                        description = item.get('description', '')
                        if isinstance(description, list):
                            description = ' '.join(description)[:500]
                        
                        # Create prompt-completion pair
                        prompt, completion = self.create_training_prompt(
                            title, category_name, description, price
                        )
                        
                        sample = FineTuningSample(
                            title=title,
                            description=description,
                            category=category_name,
                            price=price,
                            prompt=prompt,
                            completion=completion
                        )
                        
                        all_samples.append(sample)
                        samples_loaded += 1
                        
                        # Update category stats
                        if category_name not in self.category_stats:
                            self.category_stats[category_name] = {'count': 0, 'total_price': 0}
                        self.category_stats[category_name]['count'] += 1
                        self.category_stats[category_name]['total_price'] += price
                        
                    except Exception as e:
                        continue
                
                logger.info(f"Loaded {samples_loaded} samples from {category_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue
        
        # Shuffle and balance samples
        random.shuffle(all_samples)
        logger.info(f"Total samples loaded: {len(all_samples)}")
        
        return all_samples
    
    async def create_balanced_dataset(self, samples: List[FineTuningSample], target_size: int = 10000) -> List[FineTuningSample]:
        """Create balanced dataset across categories"""
        logger.info(f"Creating balanced dataset of size {target_size}")
        
        # Group by category
        category_samples = {}
        for sample in samples:
            category = sample.category
            if category not in category_samples:
                category_samples[category] = []
            category_samples[category].append(sample)
        
        # Calculate samples per category
        num_categories = len(category_samples)
        samples_per_category = target_size // num_categories
        
        balanced_samples = []
        for category, cat_samples in category_samples.items():
            if len(cat_samples) >= samples_per_category:
                # Sample evenly
                sampled = random.sample(cat_samples, samples_per_category)
            else:
                # Use all samples and duplicate if needed
                sampled = cat_samples * (samples_per_category // len(cat_samples) + 1)
                sampled = sampled[:samples_per_category]
            
            balanced_samples.extend(sampled)
            logger.info(f"Added {len(sampled)} samples from {category}")
        
        # Final shuffle
        random.shuffle(balanced_samples)
        
        logger.info(f"Created balanced dataset with {len(balanced_samples)} samples")
        return balanced_samples
    
    def split_dataset(self, samples: List[FineTuningSample], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[FineTuningSample], List[FineTuningSample], List[FineTuningSample]]:
        """Split dataset into train/validation/test"""
        total_samples = len(samples)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        logger.info(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    async def prepare_finetuning_data(self, max_samples: int = 10000) -> Tuple[List[FineTuningSample], List[FineTuningSample], List[FineTuningSample]]:
        """Prepare complete fine-tuning dataset"""
        logger.info("Preparing fine-tuning dataset")
        
        # Load Amazon data
        all_samples = await self.load_amazon_data_for_finetuning(max_samples)
        
        if len(all_samples) < 1000:
            logger.warning("Insufficient samples, generating synthetic data")
            all_samples.extend(self.generate_synthetic_data(1000))
        
        # Create balanced dataset
        balanced_samples = await self.create_balanced_dataset(all_samples, max_samples)
        
        # Split into train/val/test
        train_samples, val_samples, test_samples = self.split_dataset(balanced_samples)
        
        # Store for later use
        self.training_samples = train_samples
        self.validation_samples = val_samples
        self.test_samples = test_samples
        
        # Print statistics
        self.print_dataset_statistics()
        
        return train_samples, val_samples, test_samples
    
    def generate_synthetic_data(self, num_samples: int) -> List[FineTuningSample]:
        """Generate synthetic training data for augmentation"""
        logger.info(f"Generating {num_samples} synthetic samples")
        
        synthetic_samples = []
        
        # Product templates
        templates = [
            ("Premium {category} Device {i}", "High-quality {category} with advanced features"),
            ("Professional {category} System {i}", "Commercial-grade {category} for professional use"),
            ("Smart {category} Solution {i}", "Intelligent {category} with IoT connectivity"),
            ("Advanced {category} Pro {i}", "Next-generation {category} with cutting-edge technology")
        ]
        
        categories = list(self.category_mapping.keys())
        
        for i in range(num_samples):
            category = random.choice(categories)
            template, desc_template = random.choice(templates)
            
            title = template.format(category=category, i=i)
            description = desc_template.format(category=category)
            
            # Generate realistic price based on category
            base_prices = {
                "Electronics": 299.99,
                "Appliances": 199.99,
                "Automotive": 499.99,
                "Office_Products": 89.99,
                "Tools_and_Home_Improvement": 149.99,
                "Cell_Phones_and_Accessories": 399.99,
                "Toys_and_Games": 49.99,
                "Musical_Instruments": 299.99
            }
            
            base_price = base_prices.get(category, 199.99)
            price = base_price * random.uniform(0.5, 2.0)
            price = max(1.0, min(1000.0, price))
            
            # Create prompt
            prompt, completion = self.create_training_prompt(title, category, description, price)
            
            sample = FineTuningSample(
                title=title,
                description=description,
                category=category,
                price=price,
                prompt=prompt,
                completion=completion
            )
            
            synthetic_samples.append(sample)
        
        return synthetic_samples
    
    def print_dataset_statistics(self):
        """Print comprehensive dataset statistics"""
        logger.info("=== Dataset Statistics ===")
        
        # Overall stats
        all_samples = self.training_samples + self.validation_samples + self.test_samples
        logger.info(f"Total samples: {len(all_samples)}")
        logger.info(f"Train: {len(self.training_samples)}, Val: {len(self.validation_samples)}, Test: {len(self.test_samples)}")
        
        # Price statistics
        prices = [s.price for s in all_samples]
        logger.info(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        logger.info(f"Average price: ${np.mean(prices):.2f}")
        logger.info(f"Median price: ${np.median(prices):.2f}")
        
        # Category distribution
        category_counts = {}
        for sample in all_samples:
            cat = sample.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info("Category distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(all_samples)) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Prompt length statistics
        prompt_lengths = [len(s.prompt) for s in all_samples]
        logger.info(f"Prompt length: Avg {np.mean(prompt_lengths):.1f}, Max {max(prompt_lengths)}")
        
        logger.info("=== End Statistics ===")
    
    async def load_sample_data(self, max_samples: int = 1000) -> List[Dict[str, Any]]:
        """Load sample data for quick testing"""
        train_samples, _, _ = await self.prepare_finetuning_data(max_samples)
        
        # Convert to dict format
        return [sample.to_dict() for sample in train_samples]
    
    async def load_existing_training_data(self) -> List[Dict[str, Any]]:
        """Load existing training data from cache"""
        # This would load from cached files or database
        return await self.load_sample_data(1000)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        return {
            "training_samples": len(self.training_samples),
            "validation_samples": len(self.validation_samples),
            "test_samples": len(self.test_samples),
            "total_samples": len(self.training_samples) + len(self.validation_samples) + len(self.test_samples),
            "categories": list(self.category_mapping.keys()),
            "category_stats": self.category_stats,
            "last_updated": datetime.utcnow().isoformat()
        }
