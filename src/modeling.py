"""Feature preparation, model training and evaluation for the arrest-prediction task.

Target  : arrest = 1 if the case ended in an adult/juvenile arrest, else 0 (about 9% positives).
Features: only information available when the crime is *reported* (no Status-derived columns).
Models  : Logistic Regression (baseline), XGBoost (strong tabular model),
          TabPFN v2 (transformer pre-trained for tabular data, in-context learning).
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["hour", "dow", "month", "is_weekend", "vict_age", "log_report_lag", "n_crime_codes",
           "n_mo", "weapon_used", "part_1_2", "lat", "lon"]
CATEGORICAL = ["area", "crime_group", "crime_desc", "vict_sex", "vict_descent", "premise",
               "weapon_group", "mo_cluster"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "arrest"


def make_xy(df: pd.DataFrame):
    """Feature frame (categoricals as pandas 'category') and target vector."""
    X = df[FEATURES].copy()
    X["part_1_2"] = X["part_1_2"].astype(float)
    for c in CATEGORICAL:
        X[c] = X[c].astype(str).astype("category")
    return X, df[TARGET].to_numpy()


def best_f1_threshold(y_true, p):
    """Probability cut-off that maximises F1 (chosen on validation data, never on the test set)."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def metrics(y_true, p, threshold, name, seconds=None):
    """Threshold-free (ROC-AUC, PR-AUC) and thresholded (F1, precision, recall) metrics."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "model": name, "n_test": len(y_true),
        "ROC-AUC": roc_auc_score(y_true, p),
        "PR-AUC": average_precision_score(y_true, p),
        "F1": f1_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred), "threshold": threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "seconds": seconds,
    }


def bootstrap_auc_ci(y_true, p, n_boot=500, seed=0):
    """95% bootstrap confidence interval of ROC-AUC (honest uncertainty on small test sets)."""
    rng = np.random.default_rng(seed)
    y_true, p = np.asarray(y_true), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        vals.append(roc_auc_score(y_true[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- Logistic Regression
def fit_logistic(X_train, y_train, max_rows=250_000, seed=42):
    """Baseline: one-hot categoricals, median-imputed + scaled numerics, class-weighted."""
    if len(X_train) > max_rows:                      # keeps the baseline fast on a laptop
        idx = np.random.default_rng(seed).choice(len(X_train), max_rows, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train[idx]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    model = Pipeline([("pre", pre),
                      ("lr", LogisticRegression(max_iter=300, C=0.5, class_weight="balanced"))])
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------- XGBoost
def fit_xgboost(X_train, y_train, X_val, y_val, seed=42, n_estimators=600):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.08, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, tree_method="hist", enable_categorical=True,
        max_cat_to_onehot=1, eval_metric="aucpr", early_stopping_rounds=40, n_jobs=-1,
        random_state=seed)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_xgboost_small(X_train, y_train, seed=42):
    """XGBoost trained on a small context (same size as TabPFN gets) for a fair comparison."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
        n_jobs=-1, random_state=seed)
    model.fit(X_train, y_train, verbose=False)
    return model


# ---------------------------------------------------------------- TabPFN (transformer)
def _tabpfn_matrix(X: pd.DataFrame):
    """TabPFN wants numbers: categoricals -> integer codes, flagged as categorical columns."""
    Z = X.copy()
    for c in CATEGORICAL:
        Z[c] = Z[c].cat.codes.astype(float)
    cat_idx = [Z.columns.get_loc(c) for c in CATEGORICAL]
    return Z.to_numpy(dtype=float), cat_idx


def tabpfn_available():
    """True if the TabPFN package can be imported (it needs PyTorch)."""
    try:
        import tabpfn  # noqa: F401
        return True
    except Exception:
        return False


def fit_tabpfn(X_train, y_train, n_estimators=4, seed=42):
    """TabPFN v2: a transformer pre-trained on millions of synthetic tables.

    `fit` does NOT run gradient descent - it stores the training rows as *context*; the
    transformer then predicts new rows in one forward pass (in-context learning).
    Version 2 is used because its weights download without a licence/login step.
    """
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")     # we run on CPU
    from tabpfn import TabPFNClassifier

    # Re-use the category codes of the *training* frame for any later frame.
    cats = {c: X_train[c].cat.categories for c in CATEGORICAL}
    Xn, cat_idx = _tabpfn_matrix(X_train)
    kwargs = dict(device="cpu", n_estimators=n_estimators, random_state=seed,
                  categorical_features_indices=cat_idx, ignore_pretraining_limits=True)
    try:   # newer releases: explicitly request v2 (no licence / login step needed)
        from tabpfn.constants import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kwargs)
    except (ImportError, AttributeError):   # tabpfn 2.0.x: v2 is already the default model
        clf = TabPFNClassifier(**kwargs)
    clf.fit(Xn, y_train)
    clf._train_categories = cats
    return clf


def predict_tabpfn(clf, X, batch=1000):
    """Probability of arrest from TabPFN (aligns category codes with the training frame)."""
    Z = X.copy()
    for c in CATEGORICAL:
        Z[c] = pd.Categorical(Z[c].astype(str), categories=clf._train_categories[c])
    Xn, _ = _tabpfn_matrix(Z)
    Xn = np.where(np.isnan(Xn), np.nan, Xn)
    out = []
    for i in range(0, len(Xn), batch):
        out.append(clf.predict_proba(Xn[i:i + batch])[:, 1])
    return np.concatenate(out)


def timed(fn, *a, **k):
    t = time.time()
    res = fn(*a, **k)
    return res, round(time.time() - t, 1)


def precision_recall_at_prevalence(y_true, p, prevalence):
    """Flag the top `prevalence` share of cases by score; report precision/recall of that list.

    A threshold-free, easy-to-explain decision rule: 'if investigators could prioritise the
    top ~9% of reports, how many of those would end in an arrest?'
    """
    y_true, p = np.asarray(y_true), np.asarray(p)
    k = max(1, int(round(prevalence * len(p))))
    top = np.argsort(-p)[:k]
    tp = y_true[top].sum()
    return float(tp / k), float(tp / max(y_true.sum(), 1))
