import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_prepare_dataset(file_path, tokenizer, max_length=128):
    """
    Load the dataset from CSV and tokenize it for model input.
    """
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Rename columns to match standard terminology
    data = data.rename(columns={"Incorrect": "source", "Correct": "target"})

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(data)

    # Tokenize dataset
    def tokenize_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
