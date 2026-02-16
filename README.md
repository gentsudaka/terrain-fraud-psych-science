# Terrain Predicts Fraud

Rugged terrain → lower fraud rates. Japan (N=1,742 municipalities) and Korea (N=260 stations).

## Key Finding

23% fewer fraud cases per SD increase in terrain ruggedness (IRR = 0.77, p = .003).

## Data

```
data/
├── analytical_dataset.csv    # Main analysis (N=1,742)
├── japan_panel/              # 18-year panel
└── korea/                    # Korea analysis
```

## Code

```bash
pip install -r code/requirements.txt
python code/02_main_analysis.py
```

## Author

**Gen Tsudaka, PhD**  
Columbia Business School · The New School for Social Research  
[GitHub](https://github.com/gentsudaka) · [ORCID](https://orcid.org/0000-0002-8969-5763)

## License

MIT
