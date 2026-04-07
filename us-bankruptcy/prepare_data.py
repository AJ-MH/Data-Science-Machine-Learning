"""
prepare_data.py
------------------------------------
Run this ONCE before opening the notebook.

The actual target values in this CSV are:
  'alive'  → 0  (company survived)
  'failed' → 1  (company went bankrupt)

Usage (from inside project-path/):
    python prepare_data.py
"""

import gzip
import json
import os
import random

import pandas as pd

# ── Step 1: Download ──────────────────────────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/"
    "main/american_bankruptcy_dataset.csv"
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Downloading american_bankruptcy_dataset.csv from GitHub...")
try:
    df_raw = pd.read_csv(URL)
    print(f"  Success. Shape: {df_raw.shape}")
except Exception as e:
    print(f"  Download failed: {e}")
    print("  Trying local copy at data/american_bankruptcy_dataset.csv ...")
    df_raw = pd.read_csv("data/american_bankruptcy_dataset.csv")
    print(f"  Loaded local file. Shape: {df_raw.shape}")

# ── Step 2: Drop non-feature columns ─────────────────────────────────────────
# 'company_name' and 'fyear' are identifiers, not financial ratio features.
# 'Division' and 'MajorGroup' are industry codes — we keep them as categorical
# features because industry type genuinely affects bankruptcy risk.
df_raw = df_raw.drop(columns=["company_name", "fyear"])

# ── Step 3: Convert target column ────────────────────────────────────────────
# Actual values are 'alive' (0) and 'failed' (1)
print("\nConverting 'status_label': 'failed' → 1, 'alive' → 0 ...")
df_raw["Bankrupt"] = (df_raw["status_label"].str.strip() == "failed").astype(int)
df_raw = df_raw.drop(columns=["status_label"])

print(f"  Bankrupt counts:      {df_raw['Bankrupt'].value_counts().to_dict()}")
print(f"  Bankrupt proportions: {df_raw['Bankrupt'].value_counts(normalize=True).round(4).to_dict()}")

if df_raw["Bankrupt"].sum() == 0:
    raise ValueError("Still all zeros — check the status_label values above.")

# ── Step 4: Add company_id ────────────────────────────────────────────────────
df_raw = df_raw.reset_index(drop=True)
df_raw.insert(0, "company_id", range(1, len(df_raw) + 1))

print(f"\nFinal columns: {list(df_raw.columns)}")
print(f"Final shape:   {df_raw.shape}")

# ── Step 5: Build schema and metadata ────────────────────────────────────────
schema_fields = []
for col in df_raw.columns:
    dtype = df_raw[col].dtype
    if dtype in ["int64", "int32"]:
        field_type = "integer"
    elif dtype == "object":
        field_type = "string"
    else:
        field_type = "number"
    schema_fields.append({"name": col, "type": field_type})

metadata = {
    "source": (
        "Pellegrino et al. (2022). Machine Learning for Bankruptcy Prediction "
        "in the American Stock Market: Dataset and Benchmarks. "
        "Future Internet, 14(8), 244."
    ),
    "repository": "https://github.com/sowide/bankruptcy_dataset",
    "license": "CC BY 4.0",
    "exchanges": "NYSE and NASDAQ",
    "period": "1999-2018",
}

# ── Step 6: Train / test-features split ──────────────────────────────────────
random.seed(42)
test_idx = set(random.sample(range(len(df_raw)), k=int(0.15 * len(df_raw))))

df_train = df_raw[~df_raw.index.isin(test_idx)].copy()
df_test  = df_raw[ df_raw.index.isin(test_idx)].drop(columns=["Bankrupt"]).copy()

print(f"\nTrain set:         {df_train.shape}")
print(f"Test-features set: {df_test.shape}")
print(f"Bankrupt in train: {df_train['Bankrupt'].sum()} ({df_train['Bankrupt'].mean():.2%})")

# ── Step 7: Wrap and save ─────────────────────────────────────────────────────
train_payload = {
    "schema": {"fields": schema_fields},
    "metadata": metadata,
    "data": df_train.to_dict(orient="records"),
}

test_schema = [f for f in schema_fields if f["name"] != "Bankrupt"]
test_payload = {
    "schema": {"fields": test_schema},
    "metadata": metadata,
    "data": df_test.to_dict(orient="records"),
}

train_path = "data/us-bankruptcy-data.json.gz"
test_path  = "data/us-bankruptcy-data-test-features.json.gz"

with gzip.open(train_path, "wt", encoding="utf-8") as f:
    json.dump(train_payload, f)
print(f"\nSaved: {train_path}  ({len(df_train):,} records)")

with gzip.open(test_path, "wt", encoding="utf-8") as f:
    json.dump(test_payload, f)
print(f"Saved: {test_path}  ({len(df_test):,} records)")

print("\n✓ Done! Delete the old json.gz files in data/ and re-run the notebook.")
