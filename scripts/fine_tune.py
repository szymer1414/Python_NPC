from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from preprocess import preprocess_dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

# Load and preprocess dataset
tokenized_dataset = preprocess_dataset("data/dialogues.json")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/npc_dwayne",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=5
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()
