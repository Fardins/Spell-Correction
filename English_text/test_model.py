import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def correct_text(model_dir, input_text):
    """
    Use the fine-tuned model to correct grammar and spelling.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, max_length=50, num_beams=5, early_stopping=True
    )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def chat_loop(model_dir):
    """
    Chat loop for interactive spell correction. Type 'quit' to exit.
    """
    print("Welcome to the Spell Correction Chat! Type 'quit' to exit.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    while True:
        given_sentence = input("Enter a sentence: ").strip()
        if given_sentence.lower() == "quit":
            break

        # Tokenize and generate corrected text
        inputs = tokenizer(given_sentence, return_tensors="pt", truncation=True, max_length=50).to(device)
        output_ids = model.generate(
            **inputs, max_length=50, num_beams=5, early_stopping=True
        )
        corrected_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Corrected sentence: {corrected_sentence}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model_dir = "./output/Checkpoint1"
    
    # Start chat loop
    chat_loop(fine_tuned_model_dir)