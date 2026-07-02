import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = os.path.join("Reddit Mental Health Data", "Reddit Mental Health Dataset.csv")
df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
print("Cleaned dataset shape:", df.shape)

df.dropna(subset=["title", "text", "target"], inplace=True)
print("Dataset shape after removing null values:", df.shape)

df["combined_text"] = df["title"].astype(str) + " " + df["text"].astype(str)
print("Title and text columns combined successfully!")

df["label"] = df["target"].astype(int)
print("Labels assigned successfully!")
print("Unique labels:", sorted(df["label"].unique()))

label_map = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar Disorder",
    3: "Personality Disorder",
    4: "Anxiety"
}

output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

with open(os.path.join(output_folder, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("Label mapping saved to processed_data/label_map.json")
print("Label map:", label_map)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

print("Dataset split completed!")
print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

df.to_csv(os.path.join(output_folder, "processed_dataset.csv"), index=False)
train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

print("Processed files saved successfully!")
print("Saved files:")
print("  1. processed_dataset.csv")
print("  2. train.csv")
print("  3. test.csv")
print("  4. label_map.json")
