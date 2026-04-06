import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load processed dataset
df = pd.read_csv("data/processed/clean_reviews.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Create output directory
os.makedirs("results/figures", exist_ok=True)

# -----------------------------
# 1 Sentiment Distribution
# -----------------------------
plt.figure()

sns.countplot(x="sentiment", data=df)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("results/figures/sentiment_distribution.png")

plt.show()

# -----------------------------
# 2 Review Length Analysis
# -----------------------------
df["review_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))

plt.figure()

sns.histplot(df["review_length"], bins=50)

plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.savefig("results/figures/review_length_distribution.png")

plt.show()

# -----------------------------
# 3 Word Cloud
# -----------------------------
text = " ".join(df["clean_text"].dropna())

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")

plt.title("Most Frequent Words")

plt.savefig("results/figures/wordcloud.png")

plt.show()
