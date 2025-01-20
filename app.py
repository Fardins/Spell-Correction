import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBart50Tokenizer

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and tokenizers
@st.cache_resource

def load_english_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    return tokenizer, model

@st.cache_resource
def load_bangla_model(model_dir):
    tokenizer = MBart50Tokenizer.from_pretrained(model_dir, src_lang="bn_IN", tgt_lang="bn_IN")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, use_safetensors=True).to(device)
    return tokenizer, model

def correct_text(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=50).to(device)
    outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Streamlit App Configuration
st.set_page_config(page_title="Spell Correction App", layout="centered")
st.title("Spell Correction App")

# Load Models
english_model_dir = "./English_text/output/Checkpoint1"
bangla_model_dir = "./Bangla_text/Checkpoint"

english_tokenizer, english_model = load_english_model(english_model_dir)
bangla_tokenizer, bangla_model = load_bangla_model(bangla_model_dir)

# User Interface
st.sidebar.title("Language Selection")
mode = st.sidebar.radio("Choose Language for Spell Correction", ("English", "Bangla"))

if mode == "English":
    st.header("English Text Spell Correction")
    input_text = st.text_area("Enter your text here:", height=100)
    if st.button("Correct Text"):
        if input_text.strip():
            corrected_text = correct_text(input_text, english_tokenizer, english_model)
            st.success("Corrected Text:")
            st.write(corrected_text)
        else:
            st.error("Please enter some text to correct.")

elif mode == "Bangla":
    st.header("Bangla Text Spell Correction")
    input_text = st.text_area("\u098f\u0996\u09be\u09a8\u09c7 \u09ac\u09be\u0982\u09b2\u09be \u099f\u09c7\u0995\u09cd\u09b8\u099f \u09b2\u09bf\u0996\u09c1\u09a8:", height=100)
    if st.button("Correct Text"):
        if input_text.strip():
            corrected_text = correct_text(input_text, bangla_tokenizer, bangla_model)
            st.success("\u09b8\u09a0\u09bf\u0995 \u09ac\u09be\u0995\u09cd\u09af:")
            st.write(corrected_text)
        else:
            st.error("\u0985\u09ac\u09b6\u09cd\u09af \u09aa\u09cd\u09b0\u09ac\u09c7\u09b6 \u09a6\u09bf\u09a8 \u09a8\u09be\u09b9\u09b2\u09c7.")