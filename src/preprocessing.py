import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("data/raw/Reviews.csv")

print("Original dataset shape:", df.shape)

# Keep only useful columns
df = df[['Score', 'Text']]

# Convert score to sentiment
def label_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['Score'].apply(label_sentiment)

# Remove original score column
df = df.drop(columns=['Score'])

# Remove missing values
df = df.dropna()

print("\nAfter cleaning:")
print(df.head())

# Show class distribution
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Save processed file
df.to_csv("data/processed/clean_reviews.csv", index=False)

print("\nSaved cleaned dataset to data/processed/")

# ---------------------------
# Basic NLP Cleaning
# ---------------------------

# project-local nltk data folder
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")

# create folder if it doesn't exist
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# tell nltk to look here
nltk.data.path.insert(0, NLTK_DATA_DIR)

# force download into this directory
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

# load stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df["clean_text"] = df["Text"].apply(clean_text)

print("\nSample cleaned text:")
print(df[["Text", "clean_text"]].head())

# save again with clean_text column
df.to_csv("data/processed/clean_reviews.csv", index=False)

print("\nDataset updated with cleaned text")