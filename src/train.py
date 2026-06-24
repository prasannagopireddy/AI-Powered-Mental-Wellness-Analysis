import pandas as pd
import numpy as np
import torch

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ---------------------------------------------------------
# Step 1: Load train and test data
# ---------------------------------------------------------

# First run preprocess.py. It will create these files.
train_file = "processed_data/train.csv"
test_file = "processed_data/test.csv"


train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Train and test files loaded successfully!")
print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# ---------------------------------------------------------
# Step 2: Separate text and labels
# ---------------------------------------------------------

train_texts = train_df["combined_text"].tolist()
test_texts = test_df["combined_text"].tolist()

train_labels = train_df["label"].astype(int).tolist()
test_labels = test_df["label"].astype(int).tolist()

print("Text and labels separated successfully!")


# ---------------------------------------------------------
# Step 3: Load DistilBERT tokenizer
# ---------------------------------------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

print("Tokenizer loaded successfully!")


# ---------------------------------------------------------
# Step 4: Tokenize the text data
# ---------------------------------------------------------

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=256
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=256
)

print("Tokenization completed successfully!")


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


train_dataset = MentalHealthDataset(train_encodings, train_labels)
test_dataset = MentalHealthDataset(test_encodings, test_labels)

print("Dataset objects created successfully!")


# ---------------------------------------------------------
# Step 6: Load DistilBERT model
# ---------------------------------------------------------

# num_labels=5 because our project has 5 mental health classes
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
)

print("DistilBERT model loaded successfully!")


# ---------------------------------------------------------
# Step 7: Define accuracy calculation
# ---------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }


# ---------------------------------------------------------
# Step 8: Training settings
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=100
)

print("Training settings completed!")


# ---------------------------------------------------------
# Step 9: Create Trainer
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Trainer created successfully!")


# ---------------------------------------------------------
# Step 10: Train the model
# ---------------------------------------------------------

trainer.train()

print("Model training completed!")


# ---------------------------------------------------------
# Step 11: Evaluate the model
# ---------------------------------------------------------

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

print("Accuracy:", accuracy_score(test_labels, y_pred))

print("\nClassification Report:\n")
print(classification_report(test_labels, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(test_labels, y_pred))


# ---------------------------------------------------------
# Step 12: Save the best model
# ---------------------------------------------------------

trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

print("Best model saved successfully in final_model folder!")
