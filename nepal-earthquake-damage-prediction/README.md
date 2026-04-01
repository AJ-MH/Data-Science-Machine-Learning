# Nepal Earthquake Building Damage Prediction

**Predicting severe damage to buildings in Ramechhap district after the 2015 Gorkha Earthquake**

## Overview
This project uses real post-disaster survey data (~260,000 buildings) from the DrivenData competition "Richter's Predictor".  
I built a SQLite database from raw CSV files, performed data wrangling, EDA, and developed classification models to identify buildings at high risk of severe damage (damage grade 3).

Focus district: **Ramechhap**

## Technical Summary
- **Data Engineering**: Created relational SQLite database (`building_features` + `building_damage`) using Python + `sqlite3`
- **Wrangling**: Removed multicollinear features (`count_floors_pre_eq`), high-cardinality and engineered binary target (`severe_damage`)
- **EDA**: Identified strong signals from `height_percentage` and `foundation_type`
- **Modeling**: Logistic Regression → Decision Tree → Random Forest
- **Evaluation**: Accuracy + F1 Score comparison

## Model Performance

| Model                  | Train Accuracy | Test Accuracy | Test F1 Score |
|------------------------|----------------|---------------|---------------|
| Baseline               | 0.6654         | 0.6654        | -             |
| Logistic Regression    | 0.6681         | 0.6670        | 0.1909        |
| Decision Tree          | 0.7700         | 0.7655        | 0.6012        |
| **Random Forest**      | 0.6726         | 0.6652        | 0.6173        |

**Best Model**: Decision Tree (Test Accuracy: 0.7655)

## Covered in the project
- Building relational databases from raw flat files (ETL)
- SQL joins and querying
- Data cleaning and smart feature selection (multicollinearity + cardinality handling)
- Model iteration and comparison
- Interpretable ML (odds ratios) and ensemble methods

## How to Run
1. Place `train_values.csv` and `train_labels.csv` in the project folder
2. Open `nepal_earthquake_building_damage_prediction.ipynb`
3. Run cells sequentially - the database `nepal_earthquake.db` will be created automatically

## Dataset
Richter's Predictor: Modeling Earthquake Damage  
DrivenData Competition  
https://www.drivendata.org/competitions/57/nepal-earthquake/data/