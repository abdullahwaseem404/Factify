import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/factify_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

explainer = shap.Explainer(
    lambda x: model(**tokenizer(x, return_tensors="pt", padding=True, truncation=True)).logits.detach().numpy()
)

texts = ["Breaking news: something shocking happened"]

shap_values = explainer(texts)

shap.plots.text(shap_values[0])