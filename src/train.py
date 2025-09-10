import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import yaml
import os

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

n_estimators = params["train"]["n_estimators"]

# Load training data
train = pd.read_csv("data/processed/train.csv")
X_train = train.drop(columns=["target"])
y_train = train["target"]

# Train model
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print(f"✅ Model trained with n_estimators={n_estimators} and saved.")