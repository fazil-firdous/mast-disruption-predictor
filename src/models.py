"""Physics baseline and factories for the ML models."""
import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


class PhysicsBaseline:
    """B0 (SPEC 9): alarm when |Ip| falls fast or radiation is high compared with beam power.

    The two thresholds are quantiles of the training clean windows, so 'slope_quantile=0.001' means
    "an |Ip| slope lower than all but 0.1% of normal windows". Scores are 1 (rule fires) or 0.
    """

    def __init__(self, slope_quantile=0.001, ratio_quantile=0.999):
        self.slope_quantile = slope_quantile
        self.ratio_quantile = ratio_quantile

    def fit(self, X, y):
        """Set the thresholds from the training windows labelled 0. Returns self."""
        clean = X[np.asarray(y) == 0]
        self.slope_threshold_ = np.nanquantile(clean["ip_abs_slope"], self.slope_quantile)
        self.ratio_threshold_ = np.inf
        if self.ratio_quantile < 1:
            self.ratio_threshold_ = np.nanquantile(clean["prad_over_pnbi"], self.ratio_quantile)
        return self

    def predict_proba(self, X):
        """Two columns like sklearn: P(clean), P(disrupting). NaN features never fire the rule."""
        slope = X["ip_abs_slope"].to_numpy(dtype=float)
        ratio = X["prad_over_pnbi"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            fires = (slope < self.slope_threshold_) | (ratio > self.ratio_threshold_)
        score = fires.astype(float)
        return np.column_stack([1 - score, score])


def make_model(name, params, balanced=True):
    """A fresh unfitted model. name is 'baseline', 'logreg', 'forest' or 'hgb'; params come from GRIDS.

    balanced=False (hgb only) trains without class weights, so its scores are probabilities (PU learning needs that).
    """
    seed = config.RANDOM_SEED
    if name == "baseline":
        return PhysicsBaseline(**params)
    # median imputation is fitted on the training split only, and the has_<signal> flags stay as features
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "logreg":
        model = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000, random_state=seed)
        return make_pipeline(imputer, StandardScaler(), model)
    if name == "forest":
        model = RandomForestClassifier(n_estimators=config.RF_TREES, class_weight="balanced_subsample",
                                       min_samples_leaf=params["min_samples_leaf"],
                                       max_features=params["max_features"], n_jobs=-1, random_state=seed)
        return make_pipeline(imputer, model)
    if name == "hgb":
        # handles NaN natively, so no imputer
        return HistGradientBoostingClassifier(learning_rate=params["learning_rate"],
                                              max_leaf_nodes=params["max_leaf_nodes"],
                                              class_weight="balanced" if balanced else None,
                                              early_stopping=True,
                                              validation_fraction=config.HGB_VALIDATION_FRACTION,
                                              random_state=seed)
    raise ValueError(f"unknown model {name!r}")


# at most 12 combinations per model (SPEC 9)
GRIDS = {
    "baseline": {"slope_quantile": [0.01, 0.003, 0.001], "ratio_quantile": [0.99, 0.997, 0.999, 1.0]},
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "forest": {"min_samples_leaf": [5, 20, 50], "max_features": ["sqrt", 0.5]},
    "hgb": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63]},
}


def grid(name):
    """Every parameter combination for one model, as a list of dicts."""
    keys = list(GRIDS[name])
    return [dict(zip(keys, values)) for values in itertools.product(*GRIDS[name].values())]


def scores(model, X):
    """Probability of 'disruption coming' for each window."""
    return model.predict_proba(X)[:, 1]
