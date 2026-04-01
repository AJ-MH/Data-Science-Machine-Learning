# Nepal Earthquake Building Damage Prediction

**Predicting severe damage to buildings in Ramechhap district after the 2015 Gorkha Earthquake**

## Overview
End-to-end classification project using DrivenData survey data. Built SQLite database from raw CSVs, performed wrangling, EDA, and multiple models with hyperparameter tuning. Includes decision tree visualization, model comparison, and ethical/demographic analysis.

Focus district: **Ramechhap**

## Technical Summary
- **Data Engineering**: Created relational SQLite database (`building_features` + `building_damage`) using Python + `sqlite3`
- **Wrangling**: Removed multicollinear features (`count_floors_pre_eq`), high-cardinality and engineered binary target (`severe_damage`)
- **EDA**: Identified strong signals from `height_percentage` and `foundation_type`
- **Modeling**: Logistic Regression → Decision Tree → Random Forest
- **Evaluation**: Accuracy + F1 Score comparison

## Model Performance

| Model                    | Test Accuracy | Notes                          |
|--------------------------|---------------|--------------------------------|
| Baseline                 | 0.6654        | Majority class                 |
| Logistic Regression      | 0.6670        | Good interpretability          |
| **Decision Tree (tuned)**| **0.7695**    | Best overall performance       |
| Random Forest            | 0.6652        | Strong F1 score                |

**Best Model**: Decision Tree (`max_depth=12`, Test Accuracy = 0.7695)

## Covered in the project
- Building relational databases from raw flat files (ETL)
- SQL joins and querying
- Data cleaning and smart feature selection (multicollinearity + cardinality handling)
- Classification models: Logistic Regression, Decision Tree (with max_depth tuning), Random Forest
- Hyperparameter tuning and overfitting analysis
- Feature importance (Gini & odds ratios)
- Ethical considerations in disaster and demographic modeling

## How to Run
1. Place `train_values.csv` and `train_labels.csv` in the project folder
2. Open `nepal_earthquake_building_damage_prediction.ipynb`
3. Run cells sequentially - the database `nepal_earthquake.db` will be created automatically

## Dataset
Richter's Predictor: Modeling Earthquake Damage  
DrivenData Competition  
https://www.drivendata.org/competitions/57/nepal-earthquake/data/