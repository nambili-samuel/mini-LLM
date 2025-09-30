# Mini-LLM: A Minimal GPT Implementation

mini-LLM/
├── README.md
├── requirements.txt
├── mini_gpt_train.py
├── config.py
├── train.py
├── generate.py
└── examples/
    └── example_usage.py

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A minimal implementation of GPT-style language model training on a single GPU. This project demonstrates the complete pipeline from dataset preparation to text generation using Hugging Face Transformers.

**Author**: Dr. Nambili Samuel  
**Repository**: https://github.com/nambili-samuel/mini-LLM.git

## 🚀 Features

- Complete training pipeline for mini GPT model
- Single GPU training with mixed precision
- Wikitext dataset integration
- Customizable model architecture
- Text generation capabilities
- Hugging Face compatible model saving

## 📋 Step-by-Step Roadmap

1. **Environment Setup** - Install dependencies and setup Python environment
2. **Dataset Preparation** - Download and preprocess Wikitext dataset
3. **Tokenization** - Convert text to tokens using GPT-2 tokenizer
4. **Model Architecture** - Configure and initialize mini GPT model
5. **Training Loop** - Train model using Hugging Face Trainer
6. **Model Saving** - Save trained model and tokenizer
7. **Text Generation** - Generate text from trained model

## 🏗️ Model Architecture (Mini GPT)

The mini GPT model is based on GPT-2 architecture with reduced parameters:



## 3. config.py
```python
"""
Configuration file for Mini-LLM training
"""

# Model Configuration
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_positions": 128,      # Context length
    "n_embd": 256,          # Embedding dimension
    "n_layer": 4,           # Number of transformer layers
    "n_head": 4,            # Number of attention heads
    "n_inner": 1024,        # Inner dimension in FFN
    "resid_pdrop": 0.1,     # Residual dropout
    "embd_pdrop": 0.1,      # Embedding dropout
    "attn_pdrop": 0.1,      # Attention dropout
}

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-5,
    "fp16": True,
    "save_steps": 500,
    "logging_steps": 50,
}

# Dataset Configuration
DATASET_CONFIG = {
    "name": "wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "train",
    "max_length": 128,
}

