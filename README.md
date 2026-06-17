# 📰 Factify – Fake News Detection System

Factify is an end-to-end **Fake News Detection System** that combines traditional machine learning, transformer-based models, and explainable AI to classify news as **Real or Fake**.

---

## 🚀 Features

* 🧠 Transformer-based classification (BERT / DistilBERT)
* 📊 Traditional ML models (TF-IDF + Logistic Regression, Naive Bayes, SVM)
* 🧹 Advanced text preprocessing (cleaning, tokenization, lemmatization)
* 🔍 Explainability using SHAP
* 🌐 Interactive Streamlit web app
* 📈 Model evaluation metrics (Accuracy, Precision, Recall, F1)

---

## 🧠 Models Used
### 🔹 Transformer Models
* BERT / DistilBERT (HuggingFace Transformers)
* Fine-tuned for binary classification (Real vs Fake)
### 🔹 Traditional ML
* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

---

## ⚙️ Installation

```bash id="fact2"
git clone https://github.com/abdullahwaseem404/Factify.git
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Web App

```bash id="fact4"
streamlit run app.py
```

---

## 🧹 Text Preprocessing

* Lowercasing
* HTML removal (BeautifulSoup)
* URL & number removal
* Tokenization (NLTK)
* Stopword removal
* Lemmatization

---


## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## 🌐 Streamlit App

* Input news text
* Predict Real vs Fake
* Displays confidence score

---
