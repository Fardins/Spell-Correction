from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from data_preprocessing import load_and_prepare_dataset

def train_model(file_path, model_name="prithivida/grammar_error_correcter_v1", output_dir="./output/Checkpoint1"):
    """
    Fine-tune the grammar and spell correction model.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Prepare dataset
    tokenized_dataset = load_and_prepare_dataset(file_path, tokenizer)

    # Split dataset into train and validation
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model("./data/spell_correction_dataset.csv")