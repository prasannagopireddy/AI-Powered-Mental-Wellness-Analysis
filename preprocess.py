import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------

file_path = os.path.join("Reddit Mental Health Data", "Reddit Mental Health Dataset.csv")

# engine='python' and on_bad_lines='skip' help avoid CSV reading errors
df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# Step 2: Drop unnamed index column if present
# ---------------------------------------------------------

# Older exports sometimes include a leftover index column
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

print("Cleaned dataset shape:", df.shape)


# ---------------------------------------------------------
# Step 3: Remove null values
# ---------------------------------------------------------

# We need title, text, and target columns for our project
df.dropna(subset=["title", "text", "target"], inplace=True)

print("Dataset shape after removing null values:", df.shape)


# ---------------------------------------------------------
# Step 4: Combine title and text columns
# ---------------------------------------------------------

# The title gives short information and text gives detailed information
df["combined_text"] = df["title"].astype(str) + " " + df["text"].astype(str)

print("Title and text columns combined successfully!")


# ---------------------------------------------------------
# Step 5: Encode labels
# ---------------------------------------------------------

# LabelEncoder converts string class names into integer labels (0, 1, 2, ...)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["target"])

print("Labels encoded successfully!")
print("Label classes:", label_encoder.classes_)


# ---------------------------------------------------------
# Step 6: Save label mapping as JSON
# ---------------------------------------------------------

# This maps each integer label back to its class name.
# Saved so that predict.py can load it automatically instead of hardcoding.
label_map = {
    int(label_encoder.transform([cls])[0]): cls
    for cls in label_encoder.classes_
}

output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

with open(os.path.join(output_folder, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("Label mapping saved to processed_data/label_map.json")
print("Label map:", label_map)


# ---------------------------------------------------------
# Step 7: Split dataset into train and test data
# ---------------------------------------------------------

# 80% data is used for training and 20% data is used for testing
# stratify keeps class distribution balanced across both splits
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print("Dataset split completed!")
print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# ---------------------------------------------------------
# Step 8: Save processed files
# ---------------------------------------------------------

df.to_csv(os.path.join(output_folder, "processed_dataset.csv"), index=False)
train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

print("Processed files saved successfully!")
print("Saved files:")
print("  1. processed_dataset.csv")
print("  2. train.csv")
print("  3. test.csv")
print("  4. label_map.json")
