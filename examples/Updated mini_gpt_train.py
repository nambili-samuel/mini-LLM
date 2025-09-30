"""
Mini-LLM Training Script
Complete training pipeline for GPT-style language model on single GPU

Author: Dr. Nambili Samuel
Repository: https://github.com/nambili-samuel/mini-LLM.git
"""

import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def main():
    print("🚀 Starting Mini-LLM Training Pipeline")
    print("=" * 50)
    
    # --------------------------
    # Step 1: Load and prepare dataset
    # --------------------------
    print("📊 Step 1: Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Remove empty texts and very short sequences
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 10)
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    
    # --------------------------
    # Step 2: Tokenization
    # --------------------------
    print("🔤 Step 2: Tokenizing dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # required for batching

    def tokenize_function(batch):
        """Tokenize batch of texts"""
        return tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print("✅ Tokenization completed")
    
    # --------------------------
    # Step 3: Define Mini GPT Model
    # --------------------------
    print("🏗️ Step 3: Building model...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,      # Context length
        n_ctx=128,
        n_embd=256,          # Embedding dimension
        n_layer=4,           # Number of layers
        n_head=4,            # Attention heads
        n_inner=1024,        # FFN dimension
        resid_pdrop=0.1,     # Residual dropout
        embd_pdrop=0.1,      # Embedding dropout
        attn_pdrop=0.1,      # Attention dropout
    )

    model = GPT2LMHeadModel(config)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # --------------------------
    # Step 4: Training setup
    # --------------------------
    print("⚙️ Step 4: Setting up training...")
    training_args = TrainingArguments(
        output_dir="./mini-gpt",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        evaluation_strategy="no",
        prediction_loss_only=True,
        fp16=True,  # mixed precision for faster training
        gradient_accumulation_steps=2,  # simulate larger batch size
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to="tensorboard",
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --------------------------
    # Step 5: Train model
    # --------------------------
    print("🎯 Step 5: Starting training...")
    print("Training might take a while... Grab a coffee! ☕")
    
    trainer.train()
    
    # --------------------------
    # Step 6: Save model and tokenizer
    # --------------------------
    print("💾 Step 6: Saving model and tokenizer...")
    model.save_pretrained("./mini-gpt")
    tokenizer.save_pretrained("./mini-gpt")
    print("✅ Model and tokenizer saved to './mini-gpt/'")
    
    # --------------------------
    # Step 7: Text generation demo
    # --------------------------
    print("🎨 Step 7: Generating sample text...")
    
    # Reload model for generation (ensures proper setup)
    model.eval()
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology",
    ]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=60,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📝 Prompt: '{prompt}'")
        print(f"🤖 Generated: {generated_text}")
        print("-" * 80)

    print("\n🎉 Training completed successfully!")
    print("📁 Model saved in: ./mini-gpt/")
    print("🚀 You can now use your trained Mini-LLM for text generation!")

if __name__ == "__main__":
    main()