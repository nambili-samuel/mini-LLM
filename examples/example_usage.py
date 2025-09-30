"""
Example usage of Mini-LLM training and generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from generate import generate_text

def example_training():
    """Example of training the model"""
    print("Run training with: python mini_gpt_train.py")

def example_generation():
    """Example of text generation"""
    # Load trained model (assuming model is trained and saved in ./mini-gpt)
    try:
        model = GPT2LMHeadModel.from_pretrained("./mini-gpt")
        tokenizer = GPT2Tokenizer.from_pretrained("./mini-gpt")
        
        # Generate text
        prompts = [
            "Once upon a time",
            "The future of technology",
            "In a world where artificial intelligence",
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            generated = generate_text(model, tokenizer, prompt, max_length=80)
            print(f"Generated: {generated}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Model not found or error loading: {e}")
        print("Please train the model first using: python mini_gpt_train.py")

if __name__ == "__main__":
    print("Mini-LLM Example Usage")
    print("=" * 50)
    example_generation()