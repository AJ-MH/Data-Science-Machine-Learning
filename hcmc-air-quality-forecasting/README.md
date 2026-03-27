# Ho Chi Minh City — PM2.5 Air Quality Forecasting

## Overview
An end-to-end time series forecasting project predicting hourly PM2.5 air pollution
levels in Ho Chi Minh City, Vietnam, using real sensor data from six monitoring
stations across the city. The pipeline spans database ingestion, iterative wrangling
of real messy sensor data, and three progressively sophisticated forecasting models
evaluated with walk-forward validation.

## Project Structure
```
hcmc-air-quality-forecasting/
├── data/
│   └── AirQuality_hcmc.csv
├── hcmc_air_quality_forecasting.ipynb
└── README.md
```

## What's covered?
- MongoDB ingestion and querying (PyMongo) - local production-style database setup
- Time series wrangling: timezone localization, outlier removal, resampling, forward-fill imputation
- ACF/PACF analysis for model order selection
- Linear Regression with lag features (bridge from tabular ML to time series)
- AutoRegression (AR) with residual diagnostics
- ARMA with hyperparameter grid search and diagnostic plots

## Dataset
**HealthyAir - Outdoor Air Quality in Ho Chi Minh City, Vietnam**
Mendeley Data (CC BY 4.0): https://data.mendeley.com/datasets/pk6tzrjks8/1
Rakholia et al. (2022). DOI: 10.17632/pk6tzrjks8.1
52,549 hourly records - 6 stations - Feb 2021 to Jun 2022

## Models Built
| Model | Evaluation | Notes |
|-------|-----------|-------|
| Baseline (mean) | Training MAE | Reference only |
| Linear Regression (Lag-1) | Static 80/20 split | Bridge to time series |
| AutoRegression AR(24) | Walk-forward validation | Lag from PACF analysis |
| ARMA(p,q) | Walk-forward validation | Best (p,q) via grid search |

## How to Run
1. Install MongoDB Community Edition and start the service
2. Install dependencies:
```bash
   pip install pymongo statsmodels pandas matplotlib plotly seaborn pytz
```
3. Download `AirQuality_hcmc.csv` from Mendeley and place in `data/`
4. Run notebook from top

## Tools & Libraries
Python · PyMongo · pandas · statsmodels · Matplotlib · Plotly · seaborn · pytz