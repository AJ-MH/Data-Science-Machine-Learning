"""
Generates synthetic A/B test session data for TechCart e-commerce experiment.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 4000  # total sessions

# --- Experiment assignment ---
group = np.random.choice(["control", "treatment"], size=N)

# --- Conversion rates ---
# Control (old page): 12% add-to-cart rate
# Treatment (new page): 15% add-to-cart rate — a real but modest improvement
converted = np.array([
    np.random.binomial(1, 0.15 if g == "treatment" else 0.12)
    for g in group
])

# --- Session metadata ---
devices = np.random.choice(["desktop", "mobile", "tablet"], size=N, p=[0.5, 0.4, 0.1])
countries = np.random.choice(
    ["US", "CA", "GB", "AU", "DE", "FR", "NG", "IN", "VN", "BR"],
    size=N,
    p=[0.30, 0.10, 0.10, 0.07, 0.07, 0.06, 0.05, 0.08, 0.05, 0.12]
)

# --- Timestamps spread over 14 days ---
start = pd.Timestamp("2024-03-01")
timestamps = [
    start + pd.Timedelta(seconds=int(s))
    for s in np.random.uniform(0, 14 * 24 * 3600, size=N)
]

# --- Session duration (seconds): treatment page is slightly more engaging ---
session_duration = np.array([
    max(10, int(np.random.normal(185 if g == "treatment" else 160, 60)))
    for g in group
])

# --- Pages viewed ---
pages_viewed = np.random.randint(1, 8, size=N)

# --- Build DataFrame ---
df = pd.DataFrame({
    "session_id": [f"sess_{i:05d}" for i in range(N)],
    "timestamp": timestamps,
    "group": group,
    "converted": converted,
    "device": devices,
    "country": countries,
    "session_duration_sec": session_duration,
    "pages_viewed": pages_viewed
})

df = df.sort_values("timestamp").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/ab_test_sessions.csv", index=False)
print(f"Generated {len(df)} sessions → data/ab_test_sessions.csv")
print(df.head())
print(f"\nConversion rates:")
print(df.groupby("group")["converted"].mean().round(4))