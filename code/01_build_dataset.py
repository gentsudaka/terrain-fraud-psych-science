#!/usr/bin/env python3
"""
01_build_dataset.py — Verify and describe the analytical dataset.

Reads the pre-built analytical_dataset.csv and prints summary statistics
that correspond to the descriptive results in the manuscript.

The full build pipeline (raw fraud geocoding, TRI aggregation, population
merge) is documented in the Methods section. This script verifies the
final dataset's integrity and structure.

Output: Console summary of all key descriptives.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ── Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "analytical_dataset.csv"))

print("=" * 70)
print("ANALYTICAL DATASET VERIFICATION")
print("=" * 70)
print(f"\nRows:        {len(df)}")
print(f"Prefectures: {df['pref_code'].nunique()}")
print(f"Columns:     {df.shape[1]}")

# ── Key counts (manuscript Section: Geographic concentration) ────────────
n_total = len(df)
n_zero = (df["fraud_count"] == 0).sum()
n_any = (df["fraud_count"] > 0).sum()
pct_zero = n_zero / n_total * 100

print(f"\n── Fraud distribution ──")
print(f"Zero fraud:      {n_zero} / {n_total} ({pct_zero:.1f}%)")
print(f"Any fraud:       {n_any} / {n_total} ({100 - pct_zero:.1f}%)")
print(f"Total cases:     {df['fraud_count'].sum()}")
print(f"Max in one muni: {df['fraud_count'].max()}")

# Top municipalities
top5 = df.nlargest(5, "fraud_count")[["muni_name", "pref_name", "fraud_count"]]
print(f"\nTop 5 municipalities:")
for _, r in top5.iterrows():
    print(f"  {r['muni_name']} ({r['pref_name']}): {r['fraud_count']} cases")

# ── Terrain comparison: fraud vs no-fraud ────────────────────────────────
fraud_muni = df[df["fraud_count"] > 0]["TRI_within_z"]
nofrd_muni = df[df["fraud_count"] == 0]["TRI_within_z"]

d_mean = fraud_muni.mean() - nofrd_muni.mean()
pooled_sd = np.sqrt(
    ((len(fraud_muni) - 1) * fraud_muni.std() ** 2
     + (len(nofrd_muni) - 1) * nofrd_muni.std() ** 2)
    / (len(fraud_muni) + len(nofrd_muni) - 2)
)
cohens_d = d_mean / pooled_sd
t_stat, t_p = stats.ttest_ind(fraud_muni, nofrd_muni)

print(f"\n── Terrain comparison ──")
print(f"Fraud municipalities mean TRI_within_z:    {fraud_muni.mean():.3f}")
print(f"No-fraud municipalities mean TRI_within_z: {nofrd_muni.mean():.3f}")
print(f"Cohen's d: {cohens_d:.3f}")
print(f"t-test:    t = {t_stat:.3f}, p = {t_p:.4e}")

# ── Descriptive statistics table ─────────────────────────────────────────
desc_vars = {
    "fraud_count": "Fraud count",
    "tri": "TRI (municipal)",
    "TRI_within_z": "TRI within-pref (z)",
    "population": "Population",
    "fraud_rate_per_100k": "Fraud rate per 100k",
}

print(f"\n── Descriptive statistics ──")
print(f"{'Variable':<25} {'N':>6} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
print("-" * 75)
for col, label in desc_vars.items():
    s = df[col].dropna()
    print(f"{label:<25} {len(s):>6} {s.mean():>10.3f} {s.std():>10.3f} "
          f"{s.min():>10.3f} {s.max():>10.3f}")

# ── Prefecture-level null ────────────────────────────────────────────────
pref_agg = df.groupby("pref_code").agg(
    tri_mean=("tri", "mean"),
    fraud_total=("fraud_count", "sum"),
    pop_total=("population", "sum"),
).reset_index()
pref_agg["rate"] = pref_agg["fraud_total"] / pref_agg["pop_total"] * 100000
r_pref, p_pref = stats.pearsonr(pref_agg["tri_mean"], pref_agg["rate"])

print(f"\n── Between-prefecture correlation ──")
print(f"Pearson r(TRI_pref, fraud_rate): {r_pref:.3f}, p = {p_pref:.3f}")
print(f"(Expected: null — effect is within-prefecture)")

print(f"\n{'=' * 70}")
print("Dataset verification complete.")
print(f"{'=' * 70}")
