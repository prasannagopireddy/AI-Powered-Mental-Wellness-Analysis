import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)


# ---------------------------------------------------------
# Step 1: Load trained model and tokenizer
# ---------------------------------------------------------

# This folder is created after running train.py
model_path = "./final_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")


# ---------------------------------------------------------
# Step 2: Class label names
# ---------------------------------------------------------

# These labels are based on our project report
class_labels = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar Disorder",
    3: "Personality Disorder",
    4: "Anxiety"
}


# ---------------------------------------------------------
# Step 3: Take input text from user
# ---------------------------------------------------------

input_text = input("Enter the text: ")


# ---------------------------------------------------------
# Step 4: Tokenization
# ---------------------------------------------------------

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)

print("Text tokenized successfully!")


# ---------------------------------------------------------
# Step 5: Prediction
# ---------------------------------------------------------

model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()


# ---------------------------------------------------------
# Step 6: Show class label
# ---------------------------------------------------------

predicted_label = class_labels[predicted_class]

print("\nPrediction Result")
print("-----------------")
print("Predicted class number:", predicted_class)
print("Predicted class label:", predicted_label)
