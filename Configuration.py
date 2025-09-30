model_config = GPT2Config(
    vocab_size=50257,
    n_positions=128,      # Context length
    n_embd=256,          # Embedding dimension
    n_layer=4,           # Number of layers
    n_head=4,            # Attention heads
    n_inner=1024,        # FFN dimension
)