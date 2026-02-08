#!/usr/bin/env python3
"""
03_robustness.py — Permutation inference, LOPO jackknife, dose-response.

Reproduces the robustness battery (Fig. 2 in the manuscript):
  (a) Permutation null distribution (10,000 within-pref shuffles)
      → Observed r ≈ −0.088, p < 10⁻⁴
  (b) Leave-one-prefecture-out jackknife (47 drops)
      → IRR range [0.636, 0.709], all p < .001
  (c) Dose-response: TRI quintiles → fraud per 100k
      → Q1 ≈ 0.31, Q5 ≈ 0.06

Output:
  - Console: All robustness statistics
  - figures/fig2_robustness.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial, Poisson
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

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
df["pref_id"] = df["pref_id"].astype(int)
print(f"Loaded {len(df)} municipalities\n")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: NB-FE fit
# ═══════════════════════════════════════════════════════════════════════════

def nb_fe_irr(data):
    """Fit NB-FE and return (IRR, p, CI_lo, CI_hi) for TRI_within_z."""
    data = data.copy().reset_index(drop=True)
    # Re-code prefectures to contiguous integers
    pref_cat = pd.Categorical(data["pref_code"])
    pref_dum = pd.get_dummies(pref_cat, prefix="p", drop_first=True).astype(float)

    X = pd.concat([data[["TRI_within_z"]].astype(float), pref_dum], axis=1)
    X = sm.add_constant(X).astype(float)
    y = data["fraud_count"].astype(float)
    offset = data["log_offset"].astype(float)

    try:
        # Estimate alpha from Poisson residuals
        pois = GLM(y, X, family=Poisson(), offset=offset).fit(maxiter=100)
        mu = pois.mu
        lhs = ((y.values - mu) ** 2 - mu) / np.maximum(mu, 1e-10)
        alpha = max(np.sum(lhs * mu) / max(np.sum(mu ** 2), 1e-10), 0.01)

        res = GLM(y, X, family=NegativeBinomial(alpha=alpha), offset=offset).fit(
            maxiter=300, method="newton"
        )
        b = res.params["TRI_within_z"]
        se = res.bse["TRI_within_z"]
        p = res.pvalues["TRI_within_z"]
        return np.exp(b), p, np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    except Exception as e:
        print(f"    Warning: model failed — {e}")
        return np.nan, np.nan, np.nan, np.nan


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: PERMUTATION INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Permutation Inference (10,000 within-prefecture shuffles)")
print("=" * 70)

N_PERM = 10_000
rng = np.random.default_rng(42)

# Observed Spearman: TRI_within_z vs fraud_rate_per_100k (raw global)
# TRI_within_z is already demeaned by prefecture, so within-pref variation is captured
observed_r, _ = stats.spearmanr(df["TRI_within_z"].values, df["fraud_rate_per_100k"].values)
print(f"Observed Spearman r = {observed_r:.4f}")

# Permutation: shuffle TRI_within_z within each prefecture, breaking the
# within-pref TRI–fraud association while preserving between-pref structure
perm_rs = np.empty(N_PERM)
pref_indices = {pid: np.where(df["pref_id"].values == pid)[0]
                for pid in df["pref_id"].unique()}
tri_orig = df["TRI_within_z"].values.copy()
fraud_orig = df["fraud_rate_per_100k"].values.copy()

for i in range(N_PERM):
    shuffled_tri = tri_orig.copy()
    for pid, idx_arr in pref_indices.items():
        # Fancy indexing returns a copy, so shuffle via permutation
        perm_order = rng.permutation(idx_arr)
        shuffled_tri[idx_arr] = tri_orig[perm_order]
    perm_rs[i], _ = stats.spearmanr(shuffled_tri, fraud_orig)

n_extreme = np.sum(perm_rs <= observed_r)
perm_p = (n_extreme + 1) / (N_PERM + 1)

print(f"Null distribution: mean = {perm_rs.mean():.4f}, SD = {perm_rs.std():.4f}")
print(f"Extreme permutations: {n_extreme} / {N_PERM}")
print(f"Permutation p = {perm_p:.6f}")
z_obs = (observed_r - perm_rs.mean()) / max(perm_rs.std(), 1e-10)
print(f"Observed is {z_obs:.1f} SDs below null mean\n")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: LEAVE-ONE-PREFECTURE-OUT JACKKNIFE
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Leave-One-Prefecture-Out Jackknife (47 drops)")
print("=" * 70)

# Full model
full_irr, full_p, _, _ = nb_fe_irr(df)
print(f"Full model: IRR = {full_irr:.4f}, p = {full_p:.4e}\n")

pref_codes = sorted(df["pref_code"].unique())
pref_name_map = df.groupby("pref_code")["pref_name"].first().to_dict()
lopo_results = []

for pc in pref_codes:
    subset = df[df["pref_code"] != pc].copy()
    irr, p, ci_lo, ci_hi = nb_fe_irr(subset)
    lopo_results.append({
        "pref_code": pc,
        "pref_name": pref_name_map.get(pc, "?"),
        "IRR": irr,
        "p": p,
        "CI_lo": ci_lo,
        "CI_hi": ci_hi,
    })
    if not np.isnan(irr):
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        print(f"  Drop {pref_name_map.get(pc,'?'):<6}: IRR = {irr:.3f}, p = {p:.4e} {star}")

lopo_df = pd.DataFrame(lopo_results)
lopo_valid = lopo_df.dropna(subset=["IRR"])
print(f"\nValid LOPO models: {len(lopo_valid)}/47")
print(f"IRR range: [{lopo_valid['IRR'].min():.3f}, {lopo_valid['IRR'].max():.3f}]")
print(f"All p < .05: {(lopo_valid['p'] < 0.05).all()}")
print(f"All p < .001: {(lopo_valid['p'] < 0.001).all()}")

# Most influential
print(f"\nMost influential (highest IRR when dropped):")
for _, r in lopo_valid.nlargest(3, "IRR").iterrows():
    print(f"  Drop {r['pref_name']}: IRR = {r['IRR']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: DOSE-RESPONSE (TRI QUINTILES)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TEST 3: Dose-Response (TRI Quintiles)")
print("=" * 70)

df["tri_quintile"] = pd.qcut(df["TRI_within_z"], 5, labels=False) + 1
quint = df.groupby("tri_quintile").agg(
    n=("fraud_count", "count"),
    fraud_sum=("fraud_count", "sum"),
    pop_sum=("population", "sum"),
    pct_fraud=("fraud_any", "mean"),
).reset_index()
quint["rate_per_100k"] = quint["fraud_sum"] / quint["pop_sum"] * 100000
quint["se"] = np.sqrt(quint["rate_per_100k"] / quint["n"])

print(f"\n{'Quintile':<10} {'n':>6} {'Fraud/100k':>12} {'% with fraud':>14}")
print("-" * 45)
for _, r in quint.iterrows():
    print(f"Q{int(r['tri_quintile']):<9} {int(r['n']):>6} {r['rate_per_100k']:>12.2f} "
          f"{r['pct_fraud']*100:>13.1f}%")

r_trend, p_trend = stats.spearmanr(quint["tri_quintile"], quint["rate_per_100k"])
print(f"\nMonotonic trend: r = {r_trend:.2f}, p = {p_trend:.3f}")
ratio = quint["rate_per_100k"].iloc[0] / max(quint["rate_per_100k"].iloc[-1], 0.001)
print(f"Q1/Q5 ratio: {ratio:.1f}-fold")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Three-panel robustness
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nGenerating Figure 2...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ── Panel (a): Permutation null ──────────────────────────────────────────
ax = axes[0]
ax.hist(perm_rs, bins=60, color="#b0b0b0", edgecolor="white", linewidth=0.3,
        density=True, label="Null distribution")
ax.axvline(observed_r, color="#c0392b", linewidth=2, linestyle="-",
           label=f"Observed r = {observed_r:.3f}")
ax.set_xlabel("Spearman r (permuted)")
ax.set_ylabel("Density")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=12,
        fontweight="bold", va="top")
perm_label = f"p < 10⁻⁴" if perm_p < 1e-4 else f"p = {perm_p:.4f}"
ax.text(0.98, 0.85, f"{perm_label}\n({n_extreme}/{N_PERM:,})",
        transform=ax.transAxes, fontsize=9, ha="right", va="top", color="#c0392b")

# ── Panel (b): LOPO forest plot ──────────────────────────────────────────
ax = axes[1]
lopo_sorted = lopo_valid.sort_values("IRR").reset_index(drop=True)
y_pos = np.arange(len(lopo_sorted))

ax.errorbar(lopo_sorted["IRR"], y_pos,
            xerr=[lopo_sorted["IRR"] - lopo_sorted["CI_lo"],
                  lopo_sorted["CI_hi"] - lopo_sorted["IRR"]],
            fmt="o", color="#4a4a4a", markersize=3, linewidth=0.6,
            capsize=0, elinewidth=0.5)
if not np.isnan(full_irr):
    ax.axvline(full_irr, color="#c0392b", linewidth=1.2, linestyle="--",
               label=f"Full sample: {full_irr:.3f}")
ax.axvline(1.0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
ax.set_xlabel("IRR (TRI within-prefecture)")
ax.set_ylabel("Dropped prefecture (sorted)")
ax.set_yticks([])
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=12,
        fontweight="bold", va="top")
ax.text(0.98, 0.02,
        f"Range: [{lopo_sorted['IRR'].min():.3f}, {lopo_sorted['IRR'].max():.3f}]",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom")

# ── Panel (c): Dose-response quintiles ───────────────────────────────────
ax = axes[2]
q_labels = ["Q1\n(flattest)", "Q2", "Q3", "Q4", "Q5\n(most\nrugged)"]
bars = ax.bar(range(5), quint["rate_per_100k"],
              color=["#d4d4d4", "#b8b8b8", "#9c9c9c", "#787878", "#4a4a4a"],
              edgecolor="black", linewidth=0.6)
ax.set_xticks(range(5))
ax.set_xticklabels(q_labels, fontsize=8)
ax.set_xlabel("TRI quintile (within-prefecture)")
ax.set_ylabel("Fraud per 100,000")
ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, fontsize=12,
        fontweight="bold", va="top")

for i, (b, r_val) in enumerate(zip(bars, quint["rate_per_100k"])):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{r_val:.2f}", ha="center", va="bottom", fontsize=8)

ax.text(0.98, 0.85, f"r = {r_trend:.2f}, p = {p_trend:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top")

plt.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_robustness.png"))
plt.close()
print(f"  Saved figures/fig2_robustness.png")
