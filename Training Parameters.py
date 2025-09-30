training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    fp16=True,
    gradient_accumulation_steps=2,
)