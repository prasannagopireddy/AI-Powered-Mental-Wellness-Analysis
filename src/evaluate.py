import pandas as pd
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ---------------------------------------------------------
# Step 1: Load test dataset
# ---------------------------------------------------------

# This file is created after running preprocess.py
test_file = "processed_data/test.csv"

test_df = pd.read_csv(test_file)

print("Test dataset loaded successfully!")
print("Testing data shape:", test_df.shape)


# ---------------------------------------------------------
# Step 2: Separate text and labels
# ---------------------------------------------------------

test_texts = test_df["combined_text"].tolist()
test_labels = test_df["label"].astype(int).tolist()

print("Test text and labels separated successfully!")


# ---------------------------------------------------------
# Step 3: Load trained DistilBERT model and tokenizer
# ---------------------------------------------------------

# This folder is created after running train.py
model_path = "./final_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

print("Trained model and tokenizer loaded successfully!")


# ---------------------------------------------------------
# Step 4: Tokenize test data
# ---------------------------------------------------------

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=256
)

print("Test data tokenized successfully!")


# ---------------------------------------------------------
# Step 5: Create Dataset class
# ---------------------------------------------------------

class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


test_dataset = MentalHealthDataset(test_encodings, test_labels)

print("Test dataset object created successfully!")


# ---------------------------------------------------------
# Step 6: Create Trainer for prediction
# ---------------------------------------------------------

eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=eval_args
)

print("Trainer created successfully!")


# ---------------------------------------------------------
# Step 7: Predict on test data
# ---------------------------------------------------------

predictions = trainer.predict(test_dataset)

# The model gives logits, so we convert them into class numbers
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_labels

print("Prediction completed successfully!")


# ---------------------------------------------------------
# Step 8: Calculate evaluation metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\nEvaluation Results")
print("------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ---------------------------------------------------------
# Step 9: Print classification report
# ---------------------------------------------------------

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))


# ---------------------------------------------------------
# Step 10: Print confusion matrix
# ---------------------------------------------------------

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
