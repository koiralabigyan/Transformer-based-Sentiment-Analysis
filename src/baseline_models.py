import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("data/processed/clean_reviews.csv")

X = df["clean_text"]
y = df["sentiment"]

# Check for missing values
print("Missing values in X:", X.isna().sum())

# Fill missing values
X = X.fillna('')

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Prepare results storage
results = []

# Create output folders
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# ======================================================
# Logistic Regression
# ======================================================

model = LogisticRegression(max_iter=200)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(y_test, predictions)

print("\nLogistic Regression Results")
print("Accuracy:", lr_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Convert report to dictionary
lr_report = classification_report(y_test, predictions, output_dict=True)

# Save metrics
results.append({
    "model": "Logistic Regression",
    "accuracy": lr_accuracy,
    "precision_macro": lr_report["macro avg"]["precision"],
    "recall_macro": lr_report["macro avg"]["recall"],
    "f1_macro": lr_report["macro avg"]["f1-score"]
})

# -----------------------------
# Confusion Matrix
# -----------------------------
cm_lr = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=model.classes_
)

disp.plot()

plt.title("Logistic Regression Confusion Matrix")

plt.savefig("results/figures/logistic_regression_confusion_matrix.png")

plt.close()

# ======================================================
# Naive Bayes
# ======================================================

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_predictions = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_predictions)

print("\nNaive Bayes Results")
print("Accuracy:", nb_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, nb_predictions))

nb_report = classification_report(y_test, nb_predictions, output_dict=True)

# Save metrics
results.append({
    "model": "Naive Bayes",
    "accuracy": nb_accuracy,
    "precision_macro": nb_report["macro avg"]["precision"],
    "recall_macro": nb_report["macro avg"]["recall"],
    "f1_macro": nb_report["macro avg"]["f1-score"]
})

# -----------------------------
# Confusion Matrix
# -----------------------------
cm_nb = confusion_matrix(y_test, nb_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb,
    display_labels=nb_model.classes_
)

disp.plot()

plt.title("Naive Bayes Confusion Matrix")

plt.savefig("results/figures/naive_bayes_confusion_matrix.png")

plt.close()

# ======================================================
# Save results to CSV
# ======================================================

results_df = pd.DataFrame(results)

results_df.to_csv("results/metrics/baseline_results.csv", index=False)

print("\nResults saved to results/metrics/baseline_results.csv")
print("Confusion matrices saved to results/figures/")