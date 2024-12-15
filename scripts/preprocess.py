from datasets import Dataset
from transformers import AutoTokenizer
import json
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
def preprocess_dataset(file_path, max_length=128):
    # Load JSON file
    data = json.load(open(file_path))
    print(f"Loaded data: {data[:2]}") 
    # Tokenize data
    def preprocess(example):
        prompt = example.get("input", "")  # Use .get() to avoid crashes if key is missing
        response = example.get("output", "")
        encoded = tokenizer(
            prompt + tokenizer.eos_token + response,
            truncation=True,
            padding="max_length",
            max_length=128
    )
        encoded["labels"] = encoded["input_ids"].copy()  # Set labels equal to input_ids
        return encoded
    dataset = Dataset.from_list(data).map(preprocess)
    return dataset
