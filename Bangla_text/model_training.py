from transformers import MBartForConditionalGeneration, Trainer, TrainingArguments
import evaluate


def load_model(checkpoint="facebook/mbart-large-50"):
    """
    Loads the pre-trained MBartForConditionalGeneration model.
    """
    try:
        model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def compute_metrics(eval_preds, tokenizer):
    """
    Computes metrics for evaluation.
    """
    metric = evaluate.load("rouge")  # Example metric for text generation
    predictions, labels = eval_preds

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE metrics
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Format the results for readability
    result = {key: value.mid.fmeasure for key, value in result.items()}
    return result


def setup_trainer(
    model, tokenizer, train_dataset, test_dataset, output_dir="./Checkpoint/", logging_dir="./logs/"
):
    """
    Sets up the Trainer with necessary configurations.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,          # Where to save checkpoints
        num_train_epochs=5,             # Number of training epochs (increased for better training)
        per_device_train_batch_size=16, # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        warmup_steps=500,               # Number of warmup steps
        weight_decay=0.01,              # Strength of weight decay
        logging_dir=logging_dir,        # Directory for storing logs
        logging_steps=100,              # Log every 100 steps
        evaluation_strategy="epoch",    # Evaluate after each epoch
        save_strategy="epoch",          # Save after each epoch
        load_best_model_at_end=True,    # Load the best model after training
        metric_for_best_model="rougeL", # Use ROUGE-L as the metric
        greater_is_better=True,         # Higher values are better for this metric
        save_total_limit=2,             # Keep only the 2 most recent checkpoints
        report_to="tensorboard",        # Log to TensorBoard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),  # Pass tokenizer
    )
    return trainer


def train_model(trainer):
    """
    Trains the model using the provided trainer.
    """
    try:
        trainer.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")


def save_model_and_tokenizer(model, tokenizer, output_dir="./Checkpoint/"):
    """
    Saves the model and tokenizer checkpoints after training.
    """
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
    except Exception as e:
        print(f"Error saving model or tokenizer: {e}")
