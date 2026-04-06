
# 1. Clean GitHub Project Structure

sentiment-analysis-nlp/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv
│   │
│   └── processed/
│       └── clean_reviews.csv
│
│
├── src/
│   │
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_baseline.py
│   ├── transformer_model.py
│   │
│   └── utils.py
│
├── results/
│   │
│   ├── metrics/
│   │   └── baseline_results.csv
│   │
│   ├── figures/
│   │   └── confusion_matrix.png
│   │
│   └── models/
│       └── distilbert_model/
│
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE


---

# 2. .gitignore

Created `.gitignore`

__pycache__/
*.pyc
*.pyo
*.pyd

.env
.venv
venv/

.ipynb_checkpoints

.DS_Store

results/models/
data/raw/

*.log


This avoids pushing:
* virtual environments
* cache files
* large raw datasets
* model weights

---

# 3. requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn
torch
transformers
datasets
tqdm

---


# Sentiment Analysis using Machine Learning and Transformers

## Overview

This project implements **sentiment classification of product reviews** using both classical machine learning models and transformer-based deep learning models.

The goal is to compare:

* Traditional NLP pipelines (TF-IDF + Naive Bayes)
* Transformer architectures (DistilBERT)

---

# Project Structure

sentiment-analysis-nlp/
│
├── data/               # raw and processed datasets
├── notebooks/          # exploratory notebooks
├── src/                # source code
├── results/            # metrics, figures, models
├── docs/               # report
├── README.md
└── requirements.txt

---

# Installation

Clone the repository:
git clone https://github.com/koiralabigyan/Transformer-based-Sentiment-Analysis
cd sentiment-analysis-nlp

Create environment and install dependencies:
uv venv
uv pip install -r requirements.txt

or

pip install -r requirements.txt


---

# Dataset

The dataset contains product reviews labeled with sentiment:

| Label | Meaning  |
| ----- | -------- |
| 0     | Negative |
| 1     | Neutral  |
| 2     | Positive |

Example columns:
Text
sentiment
clean_text


---

# Pipeline

The project follows this pipeline:

Raw Data
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Baseline Model (Naive Bayes)
   ↓
Transformer Model (DistilBERT)
   ↓
Evaluation

---

# Running the Project

## 1 Preprocess Data

uv run src/preprocessing.py

Output:

data/processed/clean_reviews.csv


---

# 2 Feature Engineering

Generate TF-IDF features.

uv run src/feature_engineering.py

---

# 3 Train Baseline Model

Train and evaluate Naive Bayes.

uv run src/train_baseline.py

Outputs:

results/metrics/baseline_results.csv
results/figures/confusion_matrix.png

---

# 4 Train Transformer Model

Train DistilBERT model.

uv run src/transformer_model.py

This performs:

* tokenization
* transformer training
* evaluation

---

# Models

## Baseline Model

Multinomial Naive Bayes with TF-IDF.

Advantages:

* Fast
* Lightweight

Limitations:

* No contextual understanding

---

## Transformer Model

DistilBERT (HuggingFace Transformers)

Advantages:

* Context-aware language understanding
* Higher accuracy

---

# Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Results

The transformer model shows improved performance compared to the baseline model due to contextual embeddings.

---

# Author

Bigyan Koirala
NLP Coursework Project