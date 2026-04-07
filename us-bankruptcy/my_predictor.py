"""Reusable module for U.S. bankruptcy prediction."""
import gzip
import json
import pickle

import pandas as pd


def wrangle(filepath):
    """Load a gzip-compressed JSON file into a tidy DataFrame."""
    with gzip.open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data["data"]).set_index("company_id")
    return df


def make_predictions(data_filepath, model_filepath):
    """Generate bankruptcy predictions for new data using a saved model.

    Parameters
    ----------
    data_filepath  : str  Path to a .json.gz file of company financial features.
    model_filepath : str  Path to a .pkl file of a trained sklearn pipeline.

    Returns
    -------
    pd.Series  Bankruptcy predictions indexed by company_id.
    """
    X_new = wrangle(data_filepath)
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_new)
    return pd.Series(y_pred, index=X_new.index, name="Bankrupt")
