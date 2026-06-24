import os

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
# Step 2: Remove null values
# ---------------------------------------------------------

# We need title, text, and target columns for our project
df.dropna(subset=["title", "text", "target"], inplace=True)

print("Dataset shape after removing null values:", df.shape)


# ---------------------------------------------------------
# Step 3: Combine title and text columns
# ---------------------------------------------------------

# The title gives short information and text gives detailed information
df["combined_text"] = df["title"].astype(str) + " " + df["text"].astype(str)

print("Title and text columns combined successfully!")


# ---------------------------------------------------------
# Step 4: Encode labels
# ---------------------------------------------------------

# LabelEncoder converts labels into numbers
# Example: 0, 1, 2, 3, 4
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["target"])

print("Labels encoded successfully!")
print("Label classes:", label_encoder.classes_)


# ---------------------------------------------------------
# Step 5: Split dataset into train and test data
# ---------------------------------------------------------

# 80% data is used for training and 20% data is used for testing
# stratify is used to keep all classes balanced in train and test data
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
# Step 6: Save processed files
# ---------------------------------------------------------

output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

df.to_csv(os.path.join(output_folder, "processed_dataset.csv"), index=False)
train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

print("Processed files saved successfully!")
print("Saved files:")
print("1. processed_dataset.csv")
print("2. train.csv")
print("3. test.csv")
