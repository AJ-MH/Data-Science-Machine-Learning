# Indian Household Finance Segmentation (IHDS-II)

## Overview
Unsupervised machine learning applied to the **India Human Development Survey II (IHDS-II, 2011–12)** to segment 3,331 Indian households that were denied credit across formal and informal lending channels.

## Project Structure

| Section | Topic |
|---|---|
| 1 | Exploratory Data Analysis — age, education, urban/rural, consumption, correlation matrices |
| 2 | Two-feature K-Means clustering (`DB5` debt vs `ASSETS` index) with elbow and silhouette evaluation |
| 3 | Multi-feature clustering pipeline (trimmed variance + `SimpleImputer` + `StandardScaler` + PCA) |
| 4 | Interactive Plotly Dash web application — live radio button + slider controls |

## Covered in the Project
- Unsupervised learning (K-Means clustering)
- Feature engineering: building a `CREDIT_DIFFICULTY` binary flag from raw survey response codes (`DB8A–F`)
- Feature selection via trimmed variance
- Missing-value imputation (`SimpleImputer`)
- Dimensionality reduction (PCA)
- Model evaluation (inertia, silhouette score)
- Interactive dashboard development (Plotly Dash)
- Working with real survey microdata (TSV format, 758 columns, mixed types)

## Dataset
**India Human Development Survey II (IHDS-II), 2011–12**  
Source: ICPSR / University of Maryland & NCAER  
URL: https://www.icpsr.umich.edu/web/DSDR/studies/36151  
License: Free for academic and research use  
Total households: 42,152 | Variables: 758 | Credit-difficulty subset: 3,331

## How to Run

### Install dependencies
```bash
pip install dash plotly scipy scikit-learn pandas matplotlib seaborn
```

### Data setup
1. Download the Household file (`DS0002`, Delimited format) from [ICPSR 36151](https://www.icpsr.umich.edu/web/DSDR/studies/36151)
2. Place it at `data/IHDS_II_household.tsv`

### Sections 1–3
Run cells sequentially in Jupyter Notebook.

### Section 4 — Dash app
Run the final Dash cell. The app renders inline in the notebook output.
It is also accessible in any browser tab at:

http://127.0.0.1:8050/

The server runs as long as the cell shows `[*]`. Stop it with the Jupyter Interrupt button.