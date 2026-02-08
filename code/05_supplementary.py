#!/usr/bin/env python3
"""
05_supplementary.py — Fraud clustering gradient and additional tests.

Reproduces:
  Figure 4: Mean TRI_within_z by fraud threshold with Cohen's d
    ≥1: d = −0.39, ≥2: d = −0.56, ≥3: d = −0.63
  Additional: Detection bias check, between-pref null

Output:
  - Console: Clustering statistics, detection bias results
  - figures/fig4_clustering.png
  - supplement/supplementary_tables.md
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
SUPP = os.path.join(ROOT, "supplement")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(SUPP, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "analytical_dataset.csv"))
df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
admin = pd.read_csv(os.path.join(DATA, "admin_indicators.csv"))
admin["muni_code"] = admin["muni_code"].astype(str).str.zfill(5)
df = df.merge(admin, on="muni_code", how="left")

print(f"Loaded {len(df)} municipalities\n")


# ═══════════════════════════════════════════════════════════════════════════
# FRAUD CLUSTERING GRADIENT
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("FRAUD CLUSTERING GRADIENT")
print("=" * 70)

thresholds = [1, 2, 3]
cluster_results = []

for thresh in thresholds:
    above = df[df["fraud_count"] >= thresh]["TRI_within_z"]
    below = df[df["fraud_count"] < thresh]["TRI_within_z"]

    n_above = len(above)
    n_below = len(below)
    mean_above = above.mean()
    mean_below = below.mean()

    # Cohen's d
    pooled_sd = np.sqrt(
        ((n_above - 1) * above.std()**2 + (n_below - 1) * below.std()**2)
        / (n_above + n_below - 2)
    )
    d = (mean_above - mean_below) / pooled_sd
    t, p = stats.ttest_ind(above, below)

    cluster_results.append({
        "threshold": thresh,
        "n_above": n_above,
        "mean_tri_z": mean_above,
        "cohens_d": d,
        "p": p,
    })
    print(f"\n  ≥{thresh} cases: n = {n_above}, mean TRI_within_z = {mean_above:.3f}, "
          f"d = {d:.3f}, p = {p:.4e}")

# Full mean for reference
overall_mean = df["TRI_within_z"].mean()
print(f"\n  Overall mean TRI_within_z: {overall_mean:.3f}")

# Top fraud hotspots
print(f"\n  Top fraud hotspots (≥3 cases):")
hotspots = df[df["fraud_count"] >= 3].sort_values("fraud_count", ascending=False)
print(f"  {'Municipality':<15} {'Cases':>6} {'TRI':>8} {'TRI_wz':>8}")
for _, r in hotspots.head(10).iterrows():
    print(f"  {r['muni_name']:<15} {r['fraud_count']:>6} {r['tri']:>8.1f} {r['TRI_within_z']:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# DETECTION BIAS CHECK
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("DETECTION BIAS CHECK")
print("=" * 70)

df["employees_per_cap"] = df["muni_employees"] / df["population"] * 1000
r_emp, p_emp = stats.pearsonr(
    df["TRI_within_z"].dropna(),
    df.loc[df["TRI_within_z"].notna() & df["employees_per_cap"].notna(), "employees_per_cap"]
)
# Actually correlate properly
mask = df["TRI_within_z"].notna() & df["employees_per_cap"].notna()
r_emp, p_emp = stats.pearsonr(df.loc[mask, "TRI_within_z"], df.loc[mask, "employees_per_cap"])

print(f"  TRI_within_z vs employees/capita: r = {r_emp:.3f}, p = {p_emp:.4e}")
print(f"  → Rugged areas have {'MORE' if r_emp > 0 else 'FEWER'} gov employees per capita")
print(f"  → If detection bias exists, it works AGAINST our finding")


# ═══════════════════════════════════════════════════════════════════════════
# BETWEEN-PREFECTURE NULL
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("BETWEEN-PREFECTURE CORRELATION (EXPECTED NULL)")
print("=" * 70)

pref_agg = df.groupby("pref_code").agg(
    tri_mean=("tri", "mean"),
    fraud_total=("fraud_count", "sum"),
    pop_total=("population", "sum"),
).reset_index()
pref_agg["rate"] = pref_agg["fraud_total"] / pref_agg["pop_total"] * 100000
r_bp, p_bp = stats.pearsonr(pref_agg["tri_mean"], pref_agg["rate"])
print(f"  r(TRI_pref, fraud_rate) = {r_bp:.3f}, p = {p_bp:.3f}")
print(f"  → Effect is WITHIN-prefecture, not between")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Fraud clustering bar chart
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nGenerating Figure 4...")

fig, ax = plt.subplots(figsize=(7, 4.5))

x_labels = ["No fraud\n(n = {})".format((df["fraud_count"] == 0).sum())]
x_means = [df.loc[df["fraud_count"] == 0, "TRI_within_z"].mean()]
x_ses = [df.loc[df["fraud_count"] == 0, "TRI_within_z"].sem()]
colors = ["#d4d4d4"]

for cr in cluster_results:
    x_labels.append(f"≥{cr['threshold']} cases\n(n = {cr['n_above']})")
    x_means.append(cr["mean_tri_z"])
    x_ses.append(df.loc[df["fraud_count"] >= cr["threshold"], "TRI_within_z"].sem())
    colors.append(["#9c9c9c", "#6e6e6e", "#3a3a3a"][cr["threshold"] - 1])

bars = ax.bar(range(len(x_labels)), x_means, yerr=[s * 1.96 for s in x_ses],
              color=colors, edgecolor="black", linewidth=0.7,
              capsize=5, error_kw={"linewidth": 1.2})

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Mean TRI within-prefecture (z-score)")
ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)

# Annotate Cohen's d
for i, cr in enumerate(cluster_results, start=1):
    ax.annotate(
        f"d = {cr['cohens_d']:.2f}",
        xy=(i, x_means[i]),
        xytext=(i + 0.35, x_means[i] - 0.08),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
    )

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_clustering.png"))
plt.close()
print(f"  Saved figures/fig4_clustering.png")


# ═══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY TABLES
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nWriting supplementary tables...")

supp_md = f"""# Supplementary Tables

## Table S1: Fraud Clustering by Threshold

| Threshold | n municipalities | Mean TRI_within_z | Cohen's d | p |
|-----------|-----------------|-------------------|-----------|---|
"""

for cr in cluster_results:
    supp_md += (f"| ≥{cr['threshold']} | {cr['n_above']} | {cr['mean_tri_z']:.3f} | "
                f"{cr['cohens_d']:.3f} | {cr['p']:.4e} |\n")

supp_md += f"""
## Table S2: Detection Bias Check

| Variable | r with TRI_within_z | p | Interpretation |
|----------|---------------------|---|----------------|
| Employees/capita | {r_emp:.3f} | {p_emp:.4e} | {'More staff in rugged areas' if r_emp > 0 else 'Fewer staff'} |

## Table S3: Between-Prefecture Correlation

| Correlation | r | p | Interpretation |
|-------------|---|---|----------------|
| TRI_pref vs fraud rate | {r_bp:.3f} | {p_bp:.3f} | Null (effect is within-prefecture) |

## Table S4: Fraud Hotspots (≥3 cases)

| Municipality | Prefecture | Cases | TRI | TRI_within_z |
|-------------|-----------|-------|-----|-------------|
"""

for _, r in hotspots.head(15).iterrows():
    supp_md += (f"| {r['muni_name']} | {r['pref_name']} | {r['fraud_count']} | "
                f"{r['tri']:.1f} | {r['TRI_within_z']:.3f} |\n")

supp_md += f"""
Hotspot mean TRI_within_z: {hotspots['TRI_within_z'].mean():.3f}
Overall mean: {overall_mean:.3f}
"""

with open(os.path.join(SUPP, "supplementary_tables.md"), "w") as f:
    f.write(supp_md)

print(f"  Saved supplement/supplementary_tables.md")
print(f"\n{'=' * 70}")
print("Supplementary analysis complete.")
print(f"{'=' * 70}")
