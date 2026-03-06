"""
Data processing pipeline for Amazon product data
Enterprise-grade ETL pipeline for SteadyPrice
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import login
from tqdm import tqdm
import structlog
from datetime import datetime

from app.core.config import settings

logger = structlog.get_logger()

@dataclass
class ProductItem:
    """Product item data structure"""
    title: str
    description: Optional[str]
    category: str
    price: float
    features: Dict[str, Any]
    weight: Optional[float] = None
    brand: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'price': self.price,
            'features': self.features,
            'weight': self.weight,
            'brand': self.brand
        }

class AmazonDataPipeline:
    """Enterprise data pipeline for Amazon product data"""
    
    DATASET_MAPPING = {
        "Appliances": "raw_meta_Appliances",
        "Automotive": "raw_meta_Automotive", 
        "Electronics": "raw_meta_Electronics",
        "Office_Products": "raw_meta_Office_Products",
        "Tools_and_Home_Improvement": "raw_meta_Tools_and_Home_Improvement",
        "Cell_Phones_and_Accessories": "raw_meta_Cell_Phones_and_Accessories",
        "Toys_and_Games": "raw_meta_Toys_and_Games",
        "Musical_Instruments": "raw_meta_Musical_Instruments"
    }
    
    def __init__(self):
        self.processed_items: List[ProductItem] = []
        self.raw_data_cache = {}
        
    async def initialize(self):
        """Initialize the pipeline"""
        logger.info("Initializing Amazon data pipeline")
        
        # Login to HuggingFace if token is provided
        if settings.HF_TOKEN:
            login(settings.HF_TOKEN, add_to_git_credential=True)
            logger.info("Logged into HuggingFace")
        
        # Create data directories
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
        
    async def load_raw_data(self, categories: List[str] = None) -> Dict[str, Any]:
        """Load raw data from HuggingFace datasets"""
        if categories is None:
            categories = list(self.DATASET_MAPPING.keys())
        
        logger.info(f"Loading raw data for categories: {categories}")
        
        for category in tqdm(categories, desc="Loading datasets"):
            try:
                dataset_name = self.DATASET_MAPPING[category]
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023", 
                    dataset_name, 
                    split="full", 
                    trust_remote_code=True
                )
                
                self.raw_data_cache[category] = dataset
                logger.info(f"Loaded {len(dataset)} items for {category}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {category}: {e}")
                continue
        
        return self.raw_data_cache
    
    def parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float"""
        if not price_str:
            return None
            
        try:
            # Remove currency symbols and convert
            price_str = price_str.replace('$', '').replace(',', '').strip()
            return float(price_str)
        except (ValueError, AttributeError):
            return None
    
    def extract_features(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from raw datapoint"""
        features = {}
        
        # Extract brand
        if 'brand' in datapoint and datapoint['brand']:
            features['brand'] = datapoint['brand']
        
        # Extract features list
        if 'features' in datapoint and datapoint['features']:
            features['product_features'] = datapoint['features']
        
        # Extract technical details
        if 'details' in datapoint and datapoint['details']:
            features['technical_details'] = datapoint['details']
        
        # Extract images count
        if 'images' in datapoint and datapoint['images']:
            features['image_count'] = len(datapoint['images'])
        
        # Extract weight
        if 'weight' in datapoint and datapoint['weight']:
            try:
                weight_str = datapoint['weight']
                # Convert weight to ounces
                if 'pounds' in weight_str.lower():
                    weight = float(weight_str.split()[0]) * 16
                else:
                    weight = float(weight_str.split()[0])
                features['weight_ounces'] = weight
            except:
                pass
        
        return features
    
    def process_datapoint(self, datapoint: Dict[str, Any], category: str) -> Optional[ProductItem]:
        """Process a single datapoint into ProductItem"""
        try:
            # Parse price
            price = self.parse_price(datapoint.get('price', ''))
            if not price or price < 1 or price > 1000:
                return None  # Filter out invalid prices
            
            # Extract title and description
            title = datapoint.get('title', '').strip()
            if not title or len(title) < 10:
                return None  # Filter out short titles
            
            description = datapoint.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)
            
            # Extract features
            features = self.extract_features(datapoint)
            
            # Create ProductItem
            item = ProductItem(
                title=title,
                description=description,
                category=category,
                price=price,
                features=features,
                weight=features.get('weight_ounces'),
                brand=features.get('brand')
            )
            
            return item
            
        except Exception as e:
            logger.warning(f"Failed to process datapoint: {e}")
            return None
    
    async def process_category(self, category: str, max_items: int = None) -> List[ProductItem]:
        """Process all items in a category"""
        if category not in self.raw_data_cache:
            logger.warning(f"No data loaded for category: {category}")
            return []
        
        dataset = self.raw_data_cache[category]
        items = []
        
        logger.info(f"Processing {len(dataset)} items for {category}")
        
        for i, datapoint in enumerate(tqdm(dataset, desc=f"Processing {category}")):
            if max_items and i >= max_items:
                break
                
            item = self.process_datapoint(datapoint, category)
            if item:
                items.append(item)
        
        logger.info(f"Processed {len(items)} valid items for {category}")
        return items
    
    async def deduplicate_items(self, items: List[ProductItem]) -> List[ProductItem]:
        """Remove duplicate items based on title and description"""
        logger.info(f"Deduplicating {len(items)} items")
        
        seen_titles = set()
        seen_content = set()
        deduplicated = []
        
        for item in tqdm(items, desc="Deduplicating"):
            # Check title uniqueness
            title_lower = item.title.lower().strip()
            if title_lower in seen_titles:
                continue
            seen_titles.add(title_lower)
            
            # Check content uniqueness (title + description)
            content = f"{item.title} {item.description or ''}".lower().strip()
            if content in seen_content:
                continue
            seen_content.add(content)
            
            deduplicated.append(item)
        
        logger.info(f"Removed {len(items) - len(deduplicated)} duplicates")
        return deduplicated
    
    def create_balanced_dataset(self, items: List[ProductItem], target_size: int = 100000) -> List[ProductItem]:
        """Create a balanced dataset across categories"""
        logger.info(f"Creating balanced dataset of size {target_size}")
        
        # Group by category
        category_items = {}
        for item in items:
            if item.category not in category_items:
                category_items[item.category] = []
            category_items[item.category].append(item)
        
        # Calculate samples per category
        num_categories = len(category_items)
        samples_per_category = target_size // num_categories
        
        balanced_items = []
        for category, cat_items in category_items.items():
            if len(cat_items) >= samples_per_category:
                # Sample evenly
                sampled = np.random.choice(cat_items, samples_per_category, replace=False).tolist()
            else:
                # Use all items if less than target
                sampled = cat_items
            
            balanced_items.extend(sampled)
            logger.info(f"Added {len(sampled)} items from {category}")
        
        # Shuffle the final dataset
        np.random.shuffle(balanced_items)
        
        logger.info(f"Created balanced dataset with {len(balanced_items)} items")
        return balanced_items
    
    async def save_dataset(self, items: List[ProductItem], output_path: str):
        """Save processed dataset to disk"""
        logger.info(f"Saving {len(items)} items to {output_path}")
        
        # Convert to DataFrame
        data = [item.to_dict() for item in items]
        df = pd.DataFrame(data)
        
        # Save as JSONL
        jsonl_path = output_path.replace('.csv', '.jsonl')
        df.to_json(jsonl_path, orient='records', lines=True)
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        
        # Create HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        
        # Save dataset
        dataset_path = output_path.replace('.csv', '_dataset')
        dataset.save_to_disk(dataset_path)
        
        logger.info(f"Dataset saved to {output_path}, {jsonl_path}, and {dataset_path}")
    
    async def run_pipeline(self, 
                          categories: List[str] = None,
                          max_items_per_category: int = None,
                          target_size: int = 100000,
                          output_path: str = None) -> str:
        """Run the complete data pipeline"""
        start_time = datetime.now()
        
        if output_path is None:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            output_path = f"{settings.UPLOAD_DIR}/steadyprice_dataset_{timestamp}.csv"
        
        logger.info("Starting Amazon data pipeline")
        
        # Step 1: Load raw data
        await self.load_raw_data(categories)
        
        # Step 2: Process all categories
        all_items = []
        for category in self.raw_data_cache.keys():
            items = await self.process_category(category, max_items_per_category)
            all_items.extend(items)
        
        # Step 3: Deduplicate
        deduplicated_items = await self.deduplicate_items(all_items)
        
        # Step 4: Create balanced dataset
        balanced_items = self.create_balanced_dataset(deduplicated_items, target_size)
        
        # Step 5: Save dataset
        await self.save_dataset(balanced_items, output_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Pipeline completed in {duration:.2f} seconds")
        logger.info(f"Final dataset: {len(balanced_items)} items saved to {output_path}")
        
        return output_path

# Pipeline runner
async def run_data_pipeline():
    """Run the data pipeline"""
    pipeline = AmazonDataPipeline()
    await pipeline.initialize()
    
    output_path = await pipeline.run_pipeline(
        categories=None,  # All categories
        max_items_per_category=200000,  # Limit per category for demo
        target_size=100000,  # Target dataset size
        output_path=None  # Auto-generate path
    )
    
    return output_path

if __name__ == "__main__":
    asyncio.run(run_data_pipeline())
