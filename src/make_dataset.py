# src/make_dataset.py (run once to create raw data)
import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

os.makedirs("data/raw", exist_ok=True)
df = load_breast_cancer(as_frame=True).frame
df.to_csv("data/raw/data.csv", index=False)
print("✅ Raw data saved to data/raw/data.csv")