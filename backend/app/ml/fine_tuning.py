"""
Advanced Fine-Tuning Module for SteadyPrice Enterprise
QLoRA-based fine-tuning of Llama models for price prediction
"""

import os
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import structlog
from datetime import datetime
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset, load_dataset
from peft import PeftModel

from app.core.config import settings
from app.models.schemas import ProductCategory

logger = structlog.get_logger()

@dataclass
class FineTuningConfig:
    """Configuration for QLoRA fine-tuning"""
    base_model: str = "meta-llama/Llama-3.2-3B"
    dataset_path: str = "ed-donner/items_prompts_lite"
    output_dir: str = "./models/fine_tuned"
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # QLoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Llama
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ]

class PromptFormatter:
    """Advanced prompt formatting for price prediction"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def create_instruction_prompt(self, title: str, category: str, description: str = "", price: float = None) -> Dict[str, str]:
        """Create structured instruction prompt for price prediction"""
        
        # Build product context
        context_parts = [
            f"Product Title: {title}",
            f"Category: {category}"
        ]
        
        if description and description.strip():
            context_parts.append(f"Description: {description}")
        
        product_context = "\n".join(context_parts)
        
        # Create instruction
        instruction = (
            "You are an expert pricing analyst. Analyze the product information below "
            "and provide an accurate price prediction in US dollars. Consider factors "
            "like brand reputation, features, quality, and market positioning.\n\n"
            f"{product_context}\n\n"
            "Price Prediction: $"
        )
        
        # Create completion
        if price is not None:
            completion = f"{price:.2f}"
        else:
            completion = ""  # For inference
        
        return {
            "instruction": instruction,
            "input": "",
            "output": completion,
            "text": instruction + completion
        }
    
    def format_training_data(self, products: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format product data for instruction fine-tuning"""
        formatted_data = []
        
        for product in tqdm(products, desc="Formatting prompts"):
            prompt_data = self.create_instruction_prompt(
                title=product.get('title', ''),
                category=product.get('category', ''),
                description=product.get('description', ''),
                price=product.get('price')
            )
            formatted_data.append(prompt_data)
        
        return formatted_data

class QLoRATrainer:
    """Advanced QLoRA training system for price prediction"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.prompt_formatter = None
        
    async def initialize(self):
        """Initialize tokenizer and model"""
        logger.info("Initializing QLoRA training system")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load base model with quantization
        logger.info(f"Loading model: {self.config.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        logger.info("Applying LoRA adapters")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Initialize prompt formatter
        self.prompt_formatter = PromptFormatter(self.tokenizer)
        
        logger.info("QLoRA training system initialized successfully")
    
    def prepare_dataset(self, products: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for fine-tuning"""
        logger.info(f"Preparing dataset with {len(products)} products")
        
        # Format prompts
        formatted_data = self.prompt_formatter.format_training_data(products)
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        logger.info("Tokenizing dataset")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Setup the trainer with custom arguments"""
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard for simplicity
            fp16=False,
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Trainer setup completed")
    
    async def train(self, products: List[Dict[str, Any]], eval_split: float = 0.1):
        """Execute fine-tuning training"""
        logger.info(f"Starting QLoRA fine-tuning with {len(products)} products")
        
        # Split dataset
        eval_size = int(len(products) * eval_split)
        train_products = products[:-eval_size]
        eval_products = products[-eval_size:]
        
        logger.info(f"Train: {len(train_products)}, Eval: {len(eval_products)}")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_products)
        eval_dataset = self.prepare_dataset(eval_products)
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        logger.info("Starting training...")
        training_result = self.trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metadata
        metadata = {
            "base_model": self.config.base_model,
            "training_samples": len(train_products),
            "eval_samples": len(eval_products),
            "training_time": str(datetime.now()),
            "final_train_loss": training_result.training_loss,
            "config": self.config.__dict__
        }
        
        with open(os.path.join(self.config.output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("QLoRA fine-tuning completed successfully!")
        return training_result
    
    async def predict_price(self, title: str, category: str, description: str = "") -> Tuple[float, float]:
        """Make price prediction using fine-tuned model"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call initialize() first.")
        
        # Create prompt
        prompt_data = self.prompt_formatter.create_instruction_prompt(
            title=title,
            category=category,
            description=description,
            price=None  # No completion for inference
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt_data["instruction"],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=True
        ).to(self.model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract price
        try:
            # Find the price in the generated text
            price_str = generated_text.split("Price Prediction: $")[-1].strip()
            price = float(price_str.split()[0])
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.7, 1.0 - abs(price - 200) / 500))
            
            return price, confidence
            
        except Exception as e:
            logger.error(f"Failed to parse price from output: {e}")
            # Fallback prediction
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
            return base_price, 0.75
    
    def save_adapter(self, save_path: str):
        """Save only the LoRA adapter"""
        if self.model:
            self.model.save_pretrained(save_path)
            logger.info(f"Adapter saved to {save_path}")
    
    def load_adapter(self, adapter_path: str):
        """Load LoRA adapter onto base model"""
        if self.model:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info(f"Adapter loaded from {adapter_path}")

class FineTuningManager:
    """High-level manager for fine-tuning operations"""
    
    def __init__(self):
        self.trainer = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the fine-tuning system"""
        config = FineTuningConfig()
        self.trainer = QLoRATrainer(config)
        await self.trainer.initialize()
        self.is_initialized = True
        logger.info("Fine-tuning manager initialized")
    
    async def train_from_data(self, products: List[Dict[str, Any]]):
        """Train model from product data"""
        if not self.is_initialized:
            await self.initialize()
        
        return await self.trainer.train(products)
    
    async def predict_price(self, title: str, category: str, description: str = "") -> Tuple[float, float]:
        """Make prediction using fine-tuned model"""
        if not self.is_initialized:
            await self.initialize()
        
        return await self.trainer.predict_price(title, category, description)
