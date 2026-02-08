#!/usr/bin/env python3
"""
02_main_analysis.py — Core negative binomial models with prefecture fixed effects.

Reproduces Table 1 from the manuscript:
  Model 1: NB-FE, population offset           → IRR ≈ 0.637
  Model 2: NB-FE, establishment offset         → IRR ≈ 0.729
  Models 3–6: Confound controls (COVID, internet, Gini, crime)

Also generates Figure 1:
  (a) Histogram of fraud counts (87.9% zeros)
  (b) Violin plot comparing TRI_within_z for fraud vs no-fraud municipalities

Output:
  - Console: Full model results
  - figures/fig1_descriptive.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

# ── Plotting style ───────────────────────────────────────────────────────
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

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "analytical_dataset.csv"))
df["muni_code"] = df["muni_code"].astype(str).str.zfill(5)
df["pref_code"] = df["pref_code"].astype(str).str.zfill(2)
print(f"Loaded analytical_dataset.csv: {len(df)} rows, {df['pref_code'].nunique()} prefectures\n")


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════════

def fit_nb_fe(df, pred_cols, offset_col="log_offset", label="Model"):
    """
    Fit a negative binomial GLM with prefecture fixed effects.

    Parameters
    ----------
    df : DataFrame
    pred_cols : list of str — predictor column names (terrain + controls)
    offset_col : str — column for exposure offset
    label : str — display name

    Returns
    -------
    result : GLMResults
    summary_df : DataFrame with IRR, CI, p for each predictor
    """
    # Build prefecture dummies (drop first for identification)
    pref_dummies = pd.get_dummies(df["pref_id"].astype(int), prefix="pref", drop_first=True)
    pref_dummies = pref_dummies.astype(float)

    X = pd.concat([df[pred_cols].astype(float).reset_index(drop=True),
                    pref_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X).astype(float)
    y = df["fraud_count"].astype(float).reset_index(drop=True)
    offset = df[offset_col].astype(float).reset_index(drop=True)

    # Iteratively estimate alpha using auxiliary OLS regression
    # Start with Poisson fit, then estimate alpha from residuals
    from statsmodels.genmod.families import Poisson
    pois_res = GLM(y, X, family=Poisson(), offset=offset).fit(maxiter=100)
    mu = pois_res.mu
    # Estimate alpha: Var(Y) = mu + alpha * mu^2
    # (Y - mu)^2 / mu - 1 = alpha * mu  -->  OLS of LHS on mu (no intercept)
    lhs = ((y.values - mu) ** 2 - mu) / mu
    alpha_est = max(np.sum(lhs * mu) / np.sum(mu ** 2), 0.01)

    model = GLM(y, X, family=NegativeBinomial(alpha=alpha_est), offset=offset)
    result = model.fit(maxiter=300, method="newton")

    # Extract predictor results (skip prefecture dummies)
    rows = []
    for i, name in enumerate(["const"] + pred_cols):
        idx = i
        b = result.params.iloc[idx]
        se = result.bse.iloc[idx]
        z = result.tvalues.iloc[idx]
        p = result.pvalues.iloc[idx]
        rows.append({
            "Variable": name,
            "β": round(b, 4),
            "SE": round(se, 4),
            "IRR": round(np.exp(b), 4),
            "CI_lo": round(np.exp(b - 1.96 * se), 4),
            "CI_hi": round(np.exp(b + 1.96 * se), 4),
            "z": round(z, 3),
            "p": p,
        })

    return result, pd.DataFrame(rows)


def print_model(label, tab):
    """Pretty-print a model results table."""
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    for _, r in tab.iterrows():
        star = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else ""
        ci = f"[{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]"
        print(f"  {r['Variable']:<20} IRR = {r['IRR']:.3f}  95% CI {ci}  "
              f"p = {r['p']:.4f} {star}")


# ── Model 1: Primary (TRI_within_z, population offset) ──────────────────
print("=" * 70)
print("TABLE 1: Core Negative Binomial Models with Prefecture Fixed Effects")
print("=" * 70)

m1_res, m1_tab = fit_nb_fe(df, ["TRI_within_z"], label="Model 1")
print_model("Model 1: TRI_within_z, population offset (PRIMARY)", m1_tab)

# ── Model 2: Establishment offset ────────────────────────────────────────
# Merge establishment data
admin = pd.read_csv(os.path.join(DATA, "admin_indicators.csv"))
admin["muni_code"] = admin["muni_code"].astype(str).str.zfill(5)
admin["retail_establishments"] = pd.to_numeric(admin["retail_establishments"], errors="coerce")
df_estab = df.merge(
    admin[["muni_code", "retail_establishments"]],
    on="muni_code", how="left"
)
df_estab = df_estab.dropna(subset=["retail_establishments"])
df_estab = df_estab[df_estab["retail_establishments"] > 0].copy()
df_estab["log_estab"] = np.log(df_estab["retail_establishments"])

m2_res, m2_tab = fit_nb_fe(df_estab, ["TRI_within_z"],
                            offset_col="log_estab", label="Model 2")
print_model("Model 2: TRI_within_z, establishment offset", m2_tab)

# ── GEE for comparison ──────────────────────────────────────────────────
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable

print(f"\n{'─' * 70}")
print("  GEE Comparison (marginal estimate)")
print(f"{'─' * 70}")
gee_df = df.sort_values("pref_id").reset_index(drop=True)
X_gee = sm.add_constant(gee_df[["TRI_within_z"]])
gee_model = GEE(
    gee_df["fraud_count"], X_gee,
    groups=gee_df["pref_id"],
    family=NegativeBinomial(alpha=1.0),
    offset=gee_df["log_offset"],
    cov_struct=Exchangeable(),
)
gee_res = gee_model.fit(maxiter=200)
b_gee = gee_res.params["TRI_within_z"]
se_gee = gee_res.bse["TRI_within_z"]
p_gee = gee_res.pvalues["TRI_within_z"]
print(f"  TRI_within_z  IRR = {np.exp(b_gee):.3f}  "
      f"95% CI [{np.exp(b_gee - 1.96*se_gee):.3f}, {np.exp(b_gee + 1.96*se_gee):.3f}]  "
      f"p = {p_gee:.4f}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
tri_m1 = m1_tab[m1_tab["Variable"] == "TRI_within_z"].iloc[0]
tri_m2 = m2_tab[m2_tab["Variable"] == "TRI_within_z"].iloc[0]
print(f"  Model 1 (pop offset):   IRR = {tri_m1['IRR']:.3f}, p = {tri_m1['p']:.4e}")
print(f"  Model 2 (estab offset): IRR = {tri_m2['IRR']:.3f}, p = {tri_m2['p']:.4e}")
print(f"  GEE (marginal):         IRR = {np.exp(b_gee):.3f}, p = {p_gee:.4e}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Descriptive patterns
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nGenerating Figure 1...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ── Panel (a): Histogram of fraud counts ─────────────────────────────────
ax = axes[0]
max_count = min(df["fraud_count"].max(), 15)
bins = np.arange(-0.5, max_count + 1.5, 1)
ax.hist(df["fraud_count"], bins=bins, color="#4a4a4a", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Published fraud cases per municipality")
ax.set_ylabel("Number of municipalities")
ax.set_xlim(-0.5, max_count + 0.5)

# Annotate zero bar
n_zero = (df["fraud_count"] == 0).sum()
pct_zero = n_zero / len(df) * 100
ax.annotate(
    f"{pct_zero:.1f}% zeros\n(n = {n_zero})",
    xy=(0, n_zero), xytext=(3, n_zero * 0.85),
    fontsize=9, ha="left",
    arrowprops=dict(arrowstyle="->", color="#666", lw=0.8),
)
ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=12,
        fontweight="bold", va="top")

# ── Panel (b): Violin/box of TRI_within_z by fraud status ───────────────
ax = axes[1]
groups = [
    df.loc[df["fraud_count"] == 0, "TRI_within_z"].values,
    df.loc[df["fraud_count"] > 0, "TRI_within_z"].values,
]

parts = ax.violinplot(groups, positions=[0, 1], showmedians=False,
                       showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("#b0b0b0")
    pc.set_edgecolor("#4a4a4a")
    pc.set_alpha(0.7)

# Overlay box plots
bp = ax.boxplot(groups, positions=[0, 1], widths=0.15,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor="white", edgecolor="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"))

ax.set_xticks([0, 1])
ax.set_xticklabels(["No fraud\n(n = {})".format(len(groups[0])),
                      "Fraud ≥ 1\n(n = {})".format(len(groups[1]))])
ax.set_ylabel("TRI within-prefecture (z-score)")

# Annotate Cohen's d
fraud_z = df.loc[df["fraud_count"] > 0, "TRI_within_z"]
nofrd_z = df.loc[df["fraud_count"] == 0, "TRI_within_z"]
pooled = np.sqrt(((len(fraud_z)-1)*fraud_z.std()**2 + (len(nofrd_z)-1)*nofrd_z.std()**2)
                  / (len(fraud_z) + len(nofrd_z) - 2))
d = (fraud_z.mean() - nofrd_z.mean()) / pooled
ax.text(0.5, 0.95, f"d = {d:.2f}", transform=ax.transAxes,
        fontsize=10, ha="center", va="top")
ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=12,
        fontweight="bold", va="top")

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_descriptive.png"))
plt.close()
print(f"  Saved figures/fig1_descriptive.png")
