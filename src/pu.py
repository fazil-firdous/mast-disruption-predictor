"""Positive-unlabelled learning for the label problem (ROADMAP A1).

Under LABEL_SOURCE 'both', the windows just before an unnoted quench are labelled 0, but some of those quenches
may be real disruptions nobody wrote down. PU learning treats every window not labelled 1 as *unlabelled*
instead of negative. Method: Elkan & Noto (2008), "Learning classifiers from only positive and unlabeled data".
Assumption to keep in mind: noted disruptions are a random sample of all disruptions (SCAR). If operators
noted mainly the obvious ones, the estimate of c below is biased.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from src import models

G_MAX = 1 - 1e-6  # keeps g / (1 - g) finite
N_FOLDS = 5  # out-of-fold scores for the weights; folds are whole shots so no shot scores itself


def label_frequency(g_model, positives, columns):
    """c = P(labelled | truly positive), estimated as the mean score of held-out labelled positives (Elkan-Noto e1)."""
    return float(np.mean(models.scores(g_model, positives[columns])))


def unlabelled_weights(g, c):
    """For each unlabelled window, the probability it is really positive: (1 - c) / c * g / (1 - g), clipped to [0, 1]."""
    g = np.clip(g, 0.0, G_MAX)
    return np.clip((1 - c) / c * g / (1 - g), 0.0, 1.0)


def fit_pu(params, train, holdout, columns):
    """Fit an HGB model with Elkan-Noto weighting. Returns (model, c, weights of the training unlabelled windows).

    Step 1: g separates labelled positives from everything else. Step 2: c from holdout positives.
    Step 3: each unlabelled window enters twice, as positive with weight w and as negative with weight 1 - w.
    The weights use out-of-fold g (grouped by shot), because in-sample scores are over-confident. g is trained
    without class weights: Elkan-Noto needs g to be a probability, which balanced weights would distort
    (NOTES 2026-10-25).
    """
    s = train["label"].to_numpy()
    g_model = models.make_model("hgb", params, balanced=False).fit(train[columns], s)
    c = label_frequency(g_model, holdout[holdout["label"] == 1], columns)
    unlabelled = s == 0
    g_oof = cross_val_predict(models.make_model("hgb", params, balanced=False), train[columns], s,
                              groups=train["shot_id"],
                              cv=GroupKFold(n_splits=N_FOLDS), method="predict_proba")[:, 1]
    w = unlabelled_weights(g_oof[unlabelled], c)

    X = pd.concat([train[columns], train.loc[unlabelled, columns]], ignore_index=True)
    y = np.concatenate([s, np.ones(unlabelled.sum(), dtype=int)])
    first_copy = np.ones(len(s))
    first_copy[unlabelled] = 1 - w
    weight = np.concatenate([first_copy, w])
    model = models.make_model("hgb", params).fit(X, y, sample_weight=weight)
    return model, c, w
