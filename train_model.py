"""
Reproduces the exact pipeline from oday_emadeldin_day_13.ipynb:
  - drop duplicates, drop rows with missing target
  - pd.get_dummies(drop_first=True) for categorical columns
  - 80/20 train/test split (random_state=42)
  - StandardScaler on all features
  - LogisticRegression
Exports the fitted scaler+model plus everything the Streamlit app needs.
"""
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean (same as notebook)
# ---------------------------------------------------------------------------
df = pd.read_csv("diabetes_prediction_dataset.csv")
df.drop_duplicates(inplace=True)
df.dropna(subset=["diabetes"], inplace=True)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X = pd.get_dummies(X, drop_first=True)
FEATURE_COLUMNS = X.columns.tolist()

# ---------------------------------------------------------------------------
# 2. Split & scale (same as notebook)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Train (same as notebook)
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
fpr, tpr, _ = roc_curve(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred).tolist()

print("Metrics:", metrics)

# ---------------------------------------------------------------------------
# 4. Feature importance (logistic regression coefficients)
# ---------------------------------------------------------------------------
importances = dict(zip(FEATURE_COLUMNS, [abs(c) for c in model.coef_[0]]))

# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
joblib.dump({"model": model, "scaler": scaler, "feature_columns": FEATURE_COLUMNS},
            "diabetes_model.joblib")

artifact = {
    "model_name": "Logistic Regression",
    "metrics": metrics,
    "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    "confusion_matrix": cm,
    "feature_importances": importances,
    "feature_columns": FEATURE_COLUMNS,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "class_balance": y.value_counts(normalize=True).to_dict(),
    "gender_categories": sorted(df["gender"].unique().tolist()),
    "smoking_categories": sorted(df["smoking_history"].unique().tolist()),
}
with open("model_artifacts.json", "w") as f:
    json.dump(artifact, f, indent=2, default=str)

df.sample(n=8000, random_state=RANDOM_STATE).to_csv("diabetes_sample.csv", index=False)

print("Saved: diabetes_model.joblib, model_artifacts.json, diabetes_sample.csv")
