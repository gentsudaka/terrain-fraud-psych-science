# Flat Terrain Predicts Fraud

**Replication materials for submission to *Psychological Science***

## Abstract

Fraud clusters geographically even when enforcement is standardized. We test whether terrain ruggedness—by fragmenting populations and local information—predicts lower rule violation. Prior work shows rugged terrain predicts preferences for stronger governance; we test whether this extends to behavior. Using verified subsidy fraud from Japan's Business Continuity Support Grant (N = 1,742 municipalities), negative binomial models with prefecture fixed effects show each within-prefecture SD increase in ruggedness predicts 23% fewer fraud cases (IRR = 0.77, 95% CI [0.65, 0.91]). The gradient survives permutation inference (p < 10⁻⁴), leave-one-out replications, and dose–response tests. It attenuates 65% controlling for density and elderly share, consistent with an observability pathway. Convergent analyses show similar gradients for theft in Japan (n = 871) and crime in South Korea (n = 260); panel tests (6–18 years) show no terrain × time interactions. Ecology shapes compliance even under standardized enforcement.

## Repository Structure

```
├── README.md
├── LICENSE
├── manuscript/
│   ├── manuscript.tex              # Main manuscript (LaTeX)
│   ├── manuscript_anonymized.tex   # Anonymized version for review
│   ├── supplementary_information.tex
│   ├── cover_letter.tex
│   ├── references.bib
│   └── apa7.cls
├── figures/
│   ├── fig0_conceptual.png
│   ├── fig1_descriptive.png
│   ├── fig2_robustness.png
│   ├── fig3_opportunity.png
│   └── fig4_clustering.png
├── data/
│   ├── analytical_dataset.csv      # Main analysis file (N = 1,742)
│   ├── raw_fraud_cases.csv         # Published fraud records
│   ├── municipal_fraud.csv         # Municipality-level aggregation
│   ├── muni_tri.csv                # Terrain ruggedness by municipality
│   ├── korea/                      # South Korea replication data
│   └── japan_panel/                # Japan 18-year panel data
└── code/
    ├── 01_build_dataset.py         # Data assembly
    ├── 02_main_analysis.py         # Core models (Table 1, Figure 1)
    ├── 03_robustness.py            # Permutation, LOPO, dose-response
    ├── 04_mechanisms.py            # Mediation, moderation
    ├── 05_supplementary.py         # Additional analyses
    └── requirements.txt
```

## Replication

### Requirements
- Python ≥ 3.9
- See `code/requirements.txt` for dependencies

### Quick Start
```bash
pip install -r code/requirements.txt
python code/01_build_dataset.py
python code/02_main_analysis.py
python code/03_robustness.py
python code/04_mechanisms.py
python code/05_supplementary.py
```

## Data Sources

| Dataset | Source |
|---------|--------|
| Fraud cases | METI / SME Agency |
| Terrain (TRI) | SRTM 90m via CGIAR-CSI |
| Population | 2020 Census, e-Stat |
| Korea crime | Korean National Police Agency, data.go.kr |

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{tsudaka2026terrain,
  title   = {Flat terrain predicts fraud},
  author  = {Tsudaka, Gen},
  journal = {Psychological Science},
  year    = {2026},
  note    = {Manuscript under review}
}
```
