# U.S. Company Bankruptcy Prediction

## Overview

Can we predict which publicly listed American companies are about to go bankrupt — using only their financial ratios? This project tackles that question with data from 8,262 companies on the NYSE and NASDAQ from 1999 to 2018 (78,682 firm-year observations). The business context is real: bankruptcy prediction is a core task for credit analysts, risk managers, and regulators worldwide.

The primary challenge is **severe class imbalance** — only ~7% of observations are bankrupt companies. We address this with resampling and evaluate model quality using precision, recall, and confusion matrices rather than raw accuracy.

---

## Project Structure

```
us-bankruptcy/
│
├── data/
│   ├── us-bankruptcy-data.json.gz            # Full labelled dataset (train/validate)
│   └── us-bankruptcy-data-test-features.json.gz  # Holdout set (features only)
│
├── models/
│   ├── model-1.pkl     # Best Decision Tree (over-sampled)
│   ├── model-2.pkl     # Best Random Forest (GridSearchCV)
│   └── model-3.pkl     # Best Gradient Boosting (GridSearchCV)
│
├── us_bankruptcy_prediction.ipynb    # Main notebook (divided in 4 sections)
└── my_predictor.py                   # Reusable prediction module
```

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | Pellegrino et al. (2022), *Machine Learning for Bankruptcy Prediction in the American Stock Market*, Future Internet, MDPI |
| **Repository** | https://github.com/sowide/bankruptcy_dataset |
| **License** | CC BY 4.0 |
| **Exchanges** | NYSE and NASDAQ |
| **Period** | 1999–2018 |
| **Observations** | 78,682 firm-year records |
| **Companies** | 8,262 unique companies |
| **Features** | 18 financial ratios (leverage, liquidity, profitability, efficiency) |
| **Target** | `Bankrupt` (1 = filed Chapter 7 or Chapter 11 next fiscal year, 0 = alive) |
| **Class balance** | ~93% non-bankrupt, ~7% bankrupt |

### Key features include:
- `Debt Ratio %` — total liabilities / total assets
- `ROA(C) before interest and depreciation` — return on assets
- `Working Capital to Total Assets` — short-term liquidity indicator
- `Retained Earnings to Total Assets` — accumulated profitability
- `Current Ratio` — short-term solvency
- `Equity to Liability` — capital structure measure

---

## Addressed in the project

- CLI navigation and `gzip` decompression of data files
- Loading and parsing nested JSON into a pandas DataFrame
- Handling severely imbalanced classification data
- Random under-sampling and over-sampling (`imblearn`)
- Decision Tree classifier with Pipeline and SimpleImputer
- Confusion matrix interpretation for imbalanced problems
- Model serialisation and deserialisation with `pickle`
- `classification_report` — precision, recall, F1-score
- Random Forest ensemble with `GridSearchCV` and cross-validation
- Gradient Boosting classifier (sequential vs parallel ensembles)
- Interactive probability threshold analysis (ipywidgets)
- Business framing: regulatory recall priority vs PE firm precision priority
- Reusable prediction module (`my_predictor.py`)

---

## Model Comparison

| Model | Training Data | Notable Characteristic |
|---|---|---|
| Decision Tree (regular) | Original (imbalanced) | Near-zero recall on bankrupt class |
| Decision Tree (under-sampled) | Balanced (reduced majority) | Improved recall, reduced data |
| Decision Tree (over-sampled) | Balanced (duplicated minority) | Best DT recall, deep/overfit |
| Random Forest | Over-sampled + GridSearchCV | Ensemble reduces variance |
| Gradient Boosting | Over-sampled + GridSearchCV | Sequential learning, best minority precision |

---

## How to Run

1. Clone the repository and navigate to this folder
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn imbalanced-learn ipywidgets matplotlib seaborn
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `us_bankruptcy_prediction.ipynb` and run all cells in order

To use the prediction module directly:
```python
from my_predictor import make_predictions

predictions = make_predictions(
    data_filepath="data/us-bankruptcy-data-test-features.json.gz",
    model_filepath="models/model-3.pkl"
)
```

---

## Citation

Pellegrino, M., Lombardo, G., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022).
*Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.*
Future Internet, 14(8), 244. https://doi.org/10.3390/fi14080244

Dataset repository: https://github.com/sowide/bankruptcy_dataset
