# src/transformer_model.py
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# --------------------------
# Step 1: Load Dataset
# --------------------------
df = pd.read_csv("data/processed/clean_reviews.csv")

# Map sentiment to numeric labels
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_mapping)

print("Dataset loaded. Sample:")
print(df.head())

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# --------------------------
# Step 2: Load Tokenizer
# --------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded successfully!")

# Tokenization function with reduced max_length
def tokenize(batch):
    texts = [str(x) for x in batch["clean_text"]]  # Ensure all texts are strings
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)  # Reduced max_length

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True)
print("Tokenization successful!")

# --------------------------
# Step 3: Load Model
# --------------------------
num_labels = len(label_mapping)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)
print("Model loaded successfully!")

# --------------------------
# Step 4: Prepare Dataset for PyTorch
# --------------------------
train_test = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --------------------------
# Step 5: Training Arguments
# --------------------------
training_args = TrainingArguments(
    output_dir="results/transformer_test",
    num_train_epochs=1,                  # Only 1 epoch for testing purposes
    per_device_train_batch_size=4,       # Reduced batch size for CPU training
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # Simulate larger batch size (8 in this case)
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    learning_rate=5e-5,
    logging_dir="results/logs",
    report_to="none",                   # Disable external logging
)

# --------------------------
# Step 6: Initialize Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# --------------------------
# Step 7: Training
# --------------------------
print("Starting a short training run...")
trainer.train()
print("Partial training done!")

# --------------------------
# Step 8: Test forward pass
# --------------------------
sample_texts = [
    "I love this product!",
    "It was okay, not great.",
    "Terrible experience, never again!"
]
encodings = tokenizer(
    sample_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
)

# Perform inference (no gradients required)
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

print("Sample predictions (0=neg, 1=neu, 2=pos):", predictions.tolist())