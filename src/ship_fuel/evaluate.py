"""Common held-out metrics and condition summaries."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual, predicted) -> dict:
    """MAE/RMSE in kg/s; dimensionless R squared."""
    return {"MAE_kg_s": float(mean_absolute_error(actual, predicted)),
            "RMSE_kg_s": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def condition_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report sample counts and errors in observed sea states and speed regimes."""
    rows = []
    for name in ["regime", "wave_band"]:
        for group, part in predictions.groupby(name, observed=True):
            rows.append({"grouping": name, "condition": str(group), "n": len(part),
                         **metrics(part.actual, part.predicted)})
    return pd.DataFrame(rows)
