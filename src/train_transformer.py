import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

df = pd.read_csv("data/fake_news_dataset.csv")
df = df[["Text", "label"]]

label_map = {"Fake": 0, "Real": 1}
df["label"] = df["label"].map(label_map)

dataset = Dataset.from_pandas(df)

model_name = "distilbert-base-uncased"  # change to bert / roberta

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["Text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

training_args = TrainingArguments(
    output_dir="models/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("models/factify_model")
tokenizer.save_pretrained("models/factify_model")