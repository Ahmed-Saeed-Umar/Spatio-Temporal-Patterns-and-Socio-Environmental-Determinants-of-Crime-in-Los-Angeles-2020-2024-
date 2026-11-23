"""Small statistics helpers (scipy only) used in the statistical-analysis stage.

With ~900k rows almost every test is "significant", so every test here also reports an
EFFECT SIZE (Cramer's V, rank-biserial r, odds ratio) - the number that says whether a
difference actually matters.
"""
import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(successes, n, z=1.96):
    """Wilson 95% confidence interval for a proportion (works for small and large n)."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def chi2_association(a: pd.Series, b: pd.Series):
    """Chi-square test of independence between two categorical variables + Cramer's V."""
    table = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": v, "n": int(n)}


def cramers_v_label(v):
    """Cohen-style verbal label for Cramer's V (df*=1 thresholds are 0.1/0.3/0.5)."""
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"


def mann_whitney(x, y):
    """Mann-Whitney U test + rank-biserial correlation (effect size, -1..1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - 2 * u / (len(x) * len(y))          # sign: >0 means x tends to be smaller
    return {"U": float(u), "p_value": float(p), "rank_biserial": float(-r),
            "median_x": float(np.median(x)), "median_y": float(np.median(y)),
            "n_x": len(x), "n_y": len(y)}


def odds_ratio(exposed_event, exposed_total, unexposed_event, unexposed_total):
    """Odds ratio with a 95% CI (Woolf/log method) for a 2x2 table."""
    a, b = exposed_event, exposed_total - exposed_event
    c, d = unexposed_event, unexposed_total - unexposed_event
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    return float(orr), float(lo), float(hi)


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
