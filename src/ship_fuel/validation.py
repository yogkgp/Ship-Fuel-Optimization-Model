"""Training-only marginal and joint support for honest scenario comparisons."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def fit_support(features):
    """Fit 10-neighbour density threshold from training observations only."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputer.fit_transform(features))
    neighbors = NearestNeighbors(n_neighbors=11, n_jobs=2).fit(scaled)
    distances = neighbors.kneighbors(scaled)[0][:, -1]
    return {"imputer": imputer, "scaler": scaler, "neighbors": neighbors,
            "threshold": float(np.quantile(distances, 0.95)),
            "low": features.min(), "high": features.max(),
            "columns": list(features)}


def support_mask(features, support):
    """Reject missing, marginally novel or locally sparse feature combinations."""
    features = features[support["columns"]]
    finite = np.isfinite(features.to_numpy()).all(axis=1)
    marginal = ((features >= support["low"] - 1e-9) &
                (features <= support["high"] + 1e-9)).all(axis=1).to_numpy()
    scaled = support["scaler"].transform(support["imputer"].transform(features))
    distance = support["neighbors"].kneighbors(scaled, n_neighbors=10)[0][:, -1]
    return finite & marginal & (distance <= support["threshold"]), distance
