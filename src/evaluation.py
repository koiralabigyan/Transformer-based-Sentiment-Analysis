from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import Dataset
import torch
from sklearn.metrics import classification_report
import pandas as pd


checkpoint_path = 'results/transformer_test/checkpoint-15988' 

# Load the model 
model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path, local_files_only=True)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Load test dataset
test_dataset = Dataset.from_pandas(pd.read_csv("data/processed/clean_reviews.csv"))

# Tokenize the dataset
def tokenize(batch):
    # Ensure 'clean_text' is a list of strings 
    if isinstance(batch['clean_text'], list):
        # If it contains lists, flatten the list
        if isinstance(batch['clean_text'][0], list):
            print("Flattening clean_text")
            batch['clean_text'] = [item for sublist in batch['clean_text'] for item in sublist]

        # Check if clean_text is still not a list of strings
        if not all(isinstance(text, str) for text in batch['clean_text']):
            raise ValueError("clean_text contains non-string elements.")
        
        print("Clean text is a list of strings")
    else:
        raise ValueError("clean_text is not a list.")

    # Pass to tokenizer
    return tokenizer(batch['clean_text'], padding='max_length', truncation=True, max_length=128)

# Make sure the dataset is properly tokenized
tokenized_test_dataset = test_dataset.map(tokenize, batched=True)

# Prepare the test dataset for evaluation
test_dataset = tokenized_test_dataset  
# Perform inference on the test set
predictions = []
true_labels = []

# Put the model in evaluation mode
model.eval()

for batch in test_dataset:
    batch = {key: value.to('cpu') for key, value in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['label'].cpu().numpy())

# Compute evaluation metrics
report = classification_report(true_labels, predictions, target_names=["negative", "neutral", "positive"])
print("Evaluation Metrics:")
print(report)