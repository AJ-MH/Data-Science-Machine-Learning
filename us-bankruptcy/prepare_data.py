"""
prepare_data.py
---------------
Run this script ONCE before opening the notebook.
It downloads the US bankruptcy dataset from GitHub,
converts it to the JSON.gz format expected by the notebook,
and saves train + test-features files to the data/ folder.

Usage:
    python prepare_data.py
"""

import gzip
import json
import os
import random

import pandas as pd

# ── 1. Download the raw CSV from GitHub ──────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/"
    "main/american_bankruptcy_dataset.csv"
)

print("Downloading dataset from GitHub...")
try:
    df_raw = pd.read_csv(URL)
    print(f"Downloaded. Shape: {df_raw.shape}")
except Exception as e:
    print(f"Download failed: {e}")
    print("Please download the file manually from:")
    print("  https://github.com/sowide/bankruptcy_dataset")
    print("and place it at data/american_bankruptcy_dataset.csv")
    df_raw = pd.read_csv("data/american_bankruptcy_dataset.csv")

# ── 2. Rename target column for consistency ─────────────
# The raw file uses 'status_label' (bankrupt / alive) — convert to binary int
if "status_label" in df_raw.columns:
    df_raw["Bankrupt"] = (df_raw["status_label"] == "bankrupt").astype(int)
    df_raw = df_raw.drop(columns=["status_label"])
elif "Bankrupt?" in df_raw.columns:
    df_raw = df_raw.rename(columns={"Bankrupt?": "Bankrupt"})

# ── 3. Add a company_id index column ─────────────────────────────────────────
df_raw = df_raw.reset_index(drop=True)
df_raw.insert(0, "company_id", range(1, len(df_raw) + 1))

print(f"Columns: {list(df_raw.columns)}")
print(f"Bankrupt distribution:\n{df_raw['Bankrupt'].value_counts(normalize=True).round(4)}")

# ── 4. Build schema ───────────────────────────────────────────────────────────
schema_fields = []
for col in df_raw.columns:
    dtype = df_raw[col].dtype
    if dtype in ["int64", "int32"]:
        field_type = "integer"
    else:
        field_type = "number"
    schema_fields.append({"name": col, "type": field_type})

schema = {"fields": schema_fields}

metadata = {
    "source": "Pellegrino et al. (2022). Machine Learning for Bankruptcy Prediction "
              "in the American Stock Market: Dataset and Benchmarks. "
              "Future Internet, 14(8), 244.",
    "repository": "https://github.com/sowide/bankruptcy_dataset",
    "license": "CC BY 4.0",
    "exchanges": "NYSE and NASDAQ",
    "period": "1999-2018",
}

# ── 5. Split into train (all labelled) and test-features (no label) ──────────
random.seed(42)
test_ids = set(random.sample(range(len(df_raw)), k=int(0.15 * len(df_raw))))

df_train = df_raw[~df_raw.index.isin(test_ids)].copy()
df_test  = df_raw[df_raw.index.isin(test_ids)].drop(columns=["Bankrupt"]).copy()

# ── 6. Convert to list-of-dicts (JSON records) ───────────────────────────────
train_records = df_train.to_dict(orient="records")
test_records  = df_test.to_dict(orient="records")

# ── 7. Wrap in the schema-metadata-data envelope ─────────────────────────────
train_payload = {
    "schema": schema,
    "metadata": metadata,
    "data": train_records,
}

# For test features, exclude Bankrupt from schema
test_schema_fields = [f for f in schema_fields if f["name"] != "Bankrupt"]
test_payload = {
    "schema": {"fields": test_schema_fields},
    "metadata": metadata,
    "data": test_records,
}

# ── 8. Save as .json.gz ───────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

train_path = "data/us-bankruptcy-data.json.gz"
test_path  = "data/us-bankruptcy-data-test-features.json.gz"

with gzip.open(train_path, "wt", encoding="utf-8") as f:
    json.dump(train_payload, f)
print(f"Saved: {train_path}  ({len(train_records):,} records)")

with gzip.open(test_path, "wt", encoding="utf-8") as f:
    json.dump(test_payload, f)
print(f"Saved: {test_path}  ({len(test_records):,} records)")

print("\nAll done! You can now open us_bankruptcy_prediction.ipynb.")
