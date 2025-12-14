📰 BBC Sport NLP Text Classification

A complete Natural Language Processing (NLP) pipeline for classifying BBC Sport news articles into categories such as football, cricket, rugby, tennis, and athletics.

This project includes:

✔️ Dataset preprocessing
✔️ Text cleaning & normalization
✔️ TF-IDF vectorization
✔️ Training multiple ML models
✔️ Deep evaluation with accuracy, F1-score & confusion matrix
✔️ A prediction pipeline using saved model & vectorizer

📌 Project Overview

The BBC Sport dataset contains raw text documents organized by sport category.
The goal is to automatically classify an unseen article into the correct category using machine learning.

This project demonstrates:

End-to-end NLP workflow

Strong preprocessing & feature extraction

Classic ML algorithms for text classification

Saving/loading models for real-world use

Clean, reproducible repository structure

## 📁 Repository Structure

```text
bbc-sport-nlp-classification/
│
├── notebooks/
│   ├── 01_bbc_classification.ipynb      # Full training pipeline
│   └── 02_bbc_prediction.ipynb          # Predict new articles
│
├── data/
│   └── raw/
│       └── BBCSport/                    # Original dataset (5 sport categories)
│
├── models/
│   ├── bbc_svm_model.pkl                # Trained classifier
│   └── bbc_vectorizer.pkl               # TF-IDF vectorizer
│
├── README.md
└── .gitignore
```


🧹 1. Text Preprocessing

Each document undergoes full cleaning:

Lowercasing

Removing punctuation & digits

Tokenization

Stopword removal

Lemmatization

TF-IDF transformation

TF-IDF is used to convert each article into a numerical vector capturing the importance of words.

🤖 2. Model Training

Several machine learning algorithms were tested:

| Model               | Performance     |
| ------------------- | --------------- |
| **SVM (Linear)**    | ⭐ Best F1-score |
| Logistic Regression | Strong baseline |
| Naive Bayes         | Fast, simple    |

he best model (SVM) is exported to:

models/bbc_svm_model.pkl

Evaluation includes:

Accuracy

F1-Score (macro & weighted)

Confusion Matrix

Classification Report

🔮 3. Prediction Pipeline

The prediction notebook demonstrates how to load the trained artifacts:

import pickle

vectorizer = pickle.load(open("models/bbc_vectorizer.pkl", "rb"))
model = pickle.load(open("models/bbc_svm_model.pkl", "rb"))

text = "The striker scored twice to secure a stunning victory."
x = vectorizer.transform([text])
prediction = model.predict(x)
print(prediction)

Output example: ['football']

This shows how the model can be deployed in real applications.


📁 Saved Artifacts (Models & Vectorizer)

| File                 | Description                     |
| -------------------- | ------------------------------- |
| `bbc_svm_model.pkl`  | Trained classification model    |
| `bbc_vectorizer.pkl` | TF-IDF vocabulary & transformer |
| `/data/raw/`         | Source dataset                  |

📈 Results Summary

Excellent classification performance with SVM

Distinct confusion matrix showing strong separation between categories

TF-IDF proved effective for sports news

Prediction pipeline works on any user-provided text

🚀 How to Run

1. Clone the repository
git clone https://github.com/SaadRikAI/bbc-sport-nlp-classification.git
cd bbc-sport-nlp-classification

2. Open the training notebook

 notebooks/01_bbc_classification.ipynb

3. To run predictions
  
notebooks/02_bbc_prediction.ipynb

🎯 Skills Demonstrated

NLP preprocessing

Feature extraction (TF-IDF)

Multi-class classification

Model evaluation

Saving/loading ML artifacts

Clean ML project structuring

This repository is excellent for showcasing applied NLP skills to employers or clients.

👤 Author

Saad Rik
Machine Learning & Data Analyst
GitHub: https://github.com/SaadRikAI


