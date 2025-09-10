import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import yaml
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

n_estimators = params["train"]["n_estimators"]
max_depth = params["train"]["max_depth"]
min_samples_split = params["train"]["min_samples_split"]

train = pd.read_csv("data/processed/train.csv")
X_train = train.drop(columns=["target"])
y_train = train["target"]

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print(f"✅ Model trained with n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
