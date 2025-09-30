"""
Training utilities for Mini-LLM
"""

import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

def load_and_preprocess_dataset(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1"):
    """Load and preprocess dataset"""
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, dataset_config, split="train")
    
    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    return dataset

def setup_tokenizer():
    """Setup tokenizer"""
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def create_model(config):
    """Create model from configuration"""
    print("Creating model...")
    model_config = GPT2Config(**config)
    model = GPT2LMHeadModel(model_config)
    return model

def get_training_args(output_dir="./mini-gpt", **kwargs):
    """Get training arguments"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        **kwargs
    )
    return training_args