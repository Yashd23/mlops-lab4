import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

# Load raw
df = pd.read_csv("data/raw/data.csv")

# Train/test split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✅ Preprocessing done. Train/test saved.")