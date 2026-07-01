import json

import torch
import torch.nn.functional as F

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
model     = DistilBertForSequenceClassification.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")


# ---------------------------------------------------------
# Step 2: Load label mapping
# ---------------------------------------------------------

# label_map.json is created by preprocess.py
# It maps integer labels back to human-readable class names
with open("processed_data/label_map.json", "r") as f:
    label_map = json.load(f)

# JSON keys are always strings; convert them to integers
label_map = {int(k): v for k, v in label_map.items()}

print("Label mapping loaded:", label_map)


# ---------------------------------------------------------
# Step 3: Take input text from user
# ---------------------------------------------------------

input_text = input("\nEnter the text to analyse: ")


# ---------------------------------------------------------
# Step 4: Tokenization
# ---------------------------------------------------------

inputs = tokenizer(
    input_text,
    return_tensors = "pt",
    truncation     = True,
    padding        = True,
    max_length     = 512
)

print("Text tokenized successfully!")


# ---------------------------------------------------------
# Step 5: Prediction
# ---------------------------------------------------------

model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    logits  = outputs.logits

    # Convert logits to probabilities using softmax
    probabilities     = F.softmax(logits, dim=1)
    predicted_class   = torch.argmax(probabilities, dim=1).item()
    confidence        = probabilities[0][predicted_class].item()


# ---------------------------------------------------------
# Step 6: Show result with confidence score
# ---------------------------------------------------------

predicted_label = label_map[predicted_class]

print("\nPrediction Result")
print("-----------------")
print(f"Predicted class number : {predicted_class}")
print(f"Predicted class label  : {predicted_label}")
print(f"Confidence             : {confidence:.2%}")

print("\nAll class probabilities:")
for class_idx, prob in enumerate(probabilities[0]):
    label = label_map[class_idx]
    print(f"  {label:<25} : {prob.item():.2%}")
