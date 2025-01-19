import pandas as pd
from datasets import Dataset
from transformers import MBart50Tokenizer

def clean_text(text):
    """
    Cleans a text string by removing extra whitespace and ensuring consistent formatting.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()  # Remove leading and trailing whitespace
    text = " ".join(text.split())  # Remove extra spaces
    return text


def load_data(train_path, test_path):
    """
    Loads the train and test data from CSV files with additional cleaning steps.
    """
    try:
        # Load datasets
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # Ensure required columns exist
        required_columns = {"Input", "Target"}
        if not required_columns.issubset(train_data.columns) or not required_columns.issubset(test_data.columns):
            raise ValueError("CSV files must contain 'Input' and 'Target' columns.")

        # Clean and process text columns
        for dataset in [train_data, test_data]:
            for col in ["Input", "Target"]:
                dataset[col] = dataset[col].astype(str).apply(clean_text)

        # Drop null or empty rows
        train_data = train_data.dropna(subset=["Input", "Target"])
        test_data = test_data.dropna(subset=["Input", "Target"])
        train_data = train_data[(train_data["Input"] != "") & (train_data["Target"] != "")]
        test_data = test_data[(test_data["Input"] != "") & (test_data["Target"] != "")]

        # Remove duplicates
        train_data = train_data.drop_duplicates(subset=["Input", "Target"])
        test_data = test_data.drop_duplicates(subset=["Input", "Target"])

        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_data)
        test_dataset = Dataset.from_pandas(test_data)

        return train_dataset, test_dataset

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def preprocess_function(examples, tokenizer, max_length=128):
    """
    Preprocesses the input and target data by tokenizing.
    """
    model_inputs = tokenizer(
        examples["Input"], max_length=max_length, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["Target"], max_length=max_length, truncation=True, padding="max_length"
    ).input_ids
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
    ]  # Replace padding token with -100 for loss calculation
    return model_inputs


def preprocess_datasets(train_dataset, test_dataset, tokenizer):
    """
    Applies preprocessing to both train and test datasets.
    """
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=["Input", "Target"]
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=["Input", "Target"]
    )
    return train_dataset, test_dataset


def load_tokenizer(checkpoint="facebook/mbart-large-50"):
    """
    Loads the pre-trained MBart50 tokenizer.
    """
    try:
        tokenizer = MBart50Tokenizer.from_pretrained(checkpoint)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None
