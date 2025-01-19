import os
import logging
from data_preprocessing import load_data, preprocess_datasets, load_tokenizer
from model_training import load_model, setup_trainer, train_model, save_model_and_tokenizer


def setup_logging():
    """
    Sets up logging configuration for tracking the execution of the script.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )


def main():
    # Setup logging
    setup_logging()
    logging.info("Starting the script...")

    # Define file paths
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    output_dir = "./Checkpoint"
    logging_dir = "./logs"

    # Check file paths
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error(f"Data files not found. Ensure {train_path} and {test_path} exist.")
        return

    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        train_dataset, test_dataset = load_data(train_path, test_path)
        tokenizer = load_tokenizer()
        train_dataset, test_dataset = preprocess_datasets(train_dataset, test_dataset, tokenizer)

        # Load and set up the model
        logging.info("Loading the model...")
        model = load_model()

        if model is None:
            logging.error("Failed to load the model.")
            return

        # Setup Trainer
        logging.info("Setting up the Trainer...")
        trainer = setup_trainer(model, tokenizer, train_dataset, test_dataset, output_dir, logging_dir)

        # Train the model
        logging.info("Starting training...")
        train_model(trainer)

        # Save the model and tokenizer
        logging.info("Saving the model and tokenizer...")
        save_model_and_tokenizer(model, tokenizer, output_dir)

        logging.info("Script completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
