"""
Text generation utilities for Mini-LLM
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model_and_tokenizer(model_path):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50, **kwargs):
    """Generate text from prompt"""
    # Default generation parameters
    generation_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "pad_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs.update(kwargs)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_kwargs
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_model_and_tokenizer("./mini-gpt")
    
    prompt = "The future of artificial intelligence"
    generated = generate_text(model, tokenizer, prompt, max_length=100)
    print("Generated text:")
    print(generated)