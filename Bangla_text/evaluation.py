import torch
from transformers import AutoModelForSeq2SeqLM, MBart50Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint ="./Checkpoint"

def model_call(checkpoint):
    tokenizer = MBart50Tokenizer.from_pretrained(
        checkpoint, src_lang="bn_IN", tgt_lang="bn_IN", use_fast=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_safetensors=True)
    return tokenizer, model

def chat_loop(tokenizer, model):  # Chat loop! enter 'quit' to quit
    while True:
        given_sentence = str(input("Enter a sentence: "))
        if given_sentence.lower() == "quit":
            break
        inputs = tokenizer.encode(
            given_sentence,
            truncation=True,
            return_tensors="pt",
            max_length=len(given_sentence),
        )

        output_ids = model.generate(
            inputs,
            max_new_tokens=len(given_sentence),
            early_stopping=True,
        ).to(device)
        print(
            f"Correct sentence: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}"
        )


tokenizer, model = model_call(checkpoint)
chat_loop(tokenizer, model)
