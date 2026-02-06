# 📰 Factify

# Fake News Detection App

A machine learning-based Fake News Detection system built with **Python**, **NLTK**, and **Scikit-learn**, featuring an interactive **Streamlit web app** for real-time predictions.

Working Demo: https://www.youtube.com/watch?v=9hqxtRogcA8

## 🚀 Features
- Detects whether news text is **Real** or **Fake**
- Advanced text preprocessing (HTML removal, lemmatization, stopword filtering)
- Multiple ML models evaluated (Logistic Regression, Naive Bayes, SVM)
- Best-performing model (SVM) deployed
- Interactive and user-friendly Streamlit UI
- Visual text analysis (WordCloud, frequency plots, distributions)

## 🧠 Machine Learning Pipeline
1. Text Cleaning & Preprocessing using NLTK and BeautifulSoup  
2. Feature Extraction with **TF-IDF Vectorizer**  
3. Model Training & Evaluation  
4. Model Selection based on Accuracy & F1-score  
5. Deployment using Streamlit  

## 📊 Model Performance
| Model | Accuracy |
|------|----------|
| Logistic Regression | 98.9% |
| Naive Bayes | 95.9% |
| **SVM** | **99.7%** |

## 🛠️ Tech Stack
- Python
- Pandas
- NLTK
- BeautifulSoup
- Scikit-learn
- Streamlit
- Matplotlib & Seaborn
- WordCloud

## ▶️ How to Run the App
1. Clone the repository:
```bash
   git clone https://github.com/abdullahwaseem404/Factify.git
````
2. Install required libraries:
```bash
   pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
   streamlit run app.py
```
## ⚠️ Disclaimer
This project is for educational and research purposes only. Predictions should not be considered definitive or used as a sole source for news verification.
