import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from my_preprocessing import clean_text

MODEL_PATH = "bert-base-uncased"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="Factify", layout="centered")

st.title("📰 Factify – Fake News Detector")
st.write("Detect whether a news article is **Real or Fake** using AI")

text = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = clean_text(text)

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()

        if pred == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.write("Confidence:", probs.tolist())