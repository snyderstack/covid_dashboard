"""
Statistical modeling module for COVID-19 county outcome analysis.

Provides county-level multivariate analysis building on top of the
master county feature table (COVID outcomes joined with AHRF data).
All public functions are pure data functions — no Streamlit dependencies.

Public API:
    compute_all_correlations(df, outcome_cols, factor_cols, min_n=10)
        → DataFrame of Pearson r / Spearman ρ for every factor × outcome pair

    run_rf_feature_importance(df, outcome_col, feature_cols, n_estimators=200)
        → (importance_df, error_str)
        Requires scikit-learn. Falls back to |Pearson r| ranking if unavailable.

    run_ols_regression(df, outcome_col, predictor_cols)
        → (results_dict, error_str)
        Implemented with pure numpy + scipy; no statsmodels dependency.

    compute_resilience_scores(df, outcome_col, feature_cols, cv_folds=5)
        → (scores_df, error_str)
        Cross-validated Random Forest residuals when sklearn is available;
        cross-validated OLS residuals otherwise.

Design notes:
    - All functions return (result, error_string). error_string is None on
      success and a human-readable message on failure.
    - NaN values are handled via column-wise median imputation for model
      fitting (not for correlation, which uses pairwise complete cases).
    - Correlation p-values use the Pearson t approximation from scipy.stats.
    - OLS standard errors assume homoscedastic residuals (HC0-style SE not
      implemented; add statsmodels for HC3-robust SE in future work).
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _ss


# Factor / outcome catalogs (mirrors County Factors tab)

FACTOR_COLS: Dict[str, str] = {
    # Vaccination (CDC county-level vaccination dataset, 2020–2023)
    "Vaccination Complete (%)":      "vax_complete_pct",
    "At Least 1 Dose (%)":           "vax_dose1_pct",
    "Booster Rate (%)":              "vax_booster_pct",
    "65+ Vaccination Rate (%)":      "vax_complete_65plus_pct",
    # Healthcare access (AHRF)
    "PCP per 100k":                  "pcp_per_100k",
    "Active MDs per 100k":           "total_md_per_100k",
    "Hospital Beds per 100k":        "hospital_beds_per_100k",
    "ICU Beds per 100k":             "icu_beds_per_100k",
    "SNF Beds per 100k":             "snf_beds_per_100k",
    # Economic (AHRF)
    "Median Family Income ($)":      "median_family_income",
    "Per Capita Income ($)":         "per_capita_income",
    "Unemployment Rate (%)":         "unemployment_rate",
    "Child Poverty Rate (%)":        "child_poverty_pct",
    # Education (AHRF)
    "% Without HS Diploma":          "pct_no_hs_diploma",
    "% 4-Year College Degree":       "pct_college_4yr",
    # Demographics (AHRF)
    "Population Density (per sq mi)":"pop_density_per_sqmi",
    "% Population 65+":              "pct_pop_65plus",
    "Median Age":                    "median_age",
    "% Urban Population":            "pct_urban_pop",
    "RUCC Code (1-9)":               "rucc_code",
}

OUTCOME_COLS: Dict[str, str] = {
    "Cases per 100k":         "cases_per_100k",
    "Deaths per 100k":        "deaths_per_100k",
    "Case Fatality Rate (%)": "case_fatality_rate",
}

# Reverse-lookup: internal column name → display label
_FACTOR_LABEL = {v: k for k, v in FACTOR_COLS.items()}
_OUTCOME_LABEL = {v: k for k, v in OUTCOME_COLS.items()}


def _impute_median(X: np.ndarray) -> np.ndarray:
    """Replace NaN with column-wise medians (in-place copy)."""
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median = float(np.nanmedian(col))
            col[nan_mask] = median
            X[:, j] = col
    return X


def _ols_fit(X_with_const: np.ndarray, y: np.ndarray) -> dict:
    """
    Ordinary least squares via numpy normal equations.

    Args:
        X_with_const: Design matrix with intercept column prepended.
        y:            Response vector (length n, no NaN).

    Returns dict with keys:
        beta, se, se_hc3, t_vals, p_vals, p_vals_hc3, ci_lower, ci_upper,
        r_sq, adj_r_sq, n, p, residuals, y_hat, f_stat, f_pval
    """
    n, p = X_with_const.shape

    beta, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    y_hat     = X_with_const @ beta
    residuals = y - y_hat

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_sq = 1.0 - (1.0 - r_sq) * (n - 1) / (n - p) if n > p else np.nan

    s2 = ss_res / max(n - p, 1)
    try:
        XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 0.0))
    except np.linalg.LinAlgError:
        XtX_inv = None
        se = np.full(p, np.nan)

    # HC3 heteroscedasticity-robust standard errors (MacKinnon & White 1985):
    # Var(β) = (X'X)⁻¹ X' diag(e²/(1−h)²) X (X'X)⁻¹, with h the hat-matrix
    # diagonal. HC3 inflates each residual by its leverage, giving reliable
    # inference when residual variance is non-constant — common for skewed
    # county outcome data.
    if XtX_inv is not None:
        h = np.einsum("ij,jk,ik->i", X_with_const, XtX_inv, X_with_const)
        h = np.clip(h, 0.0, 1.0 - 1e-8)
        omega = (residuals / (1.0 - h)) ** 2
        cov_hc3 = XtX_inv @ (X_with_const.T * omega) @ X_with_const @ XtX_inv
        se_hc3 = np.sqrt(np.maximum(np.diag(cov_hc3), 0.0))
    else:
        se_hc3 = np.full(p, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_vals = beta / np.where(se > 0, se, np.nan)
        p_vals = np.where(
            np.isfinite(t_vals),
            2.0 * _ss.t.sf(np.abs(t_vals), df=max(n - p, 1)),
            np.nan,
        )
        t_hc3 = beta / np.where(se_hc3 > 0, se_hc3, np.nan)
        p_vals_hc3 = np.where(
            np.isfinite(t_hc3),
            2.0 * _ss.t.sf(np.abs(t_hc3), df=max(n - p, 1)),
            np.nan,
        )

    t_crit  = float(_ss.t.ppf(0.975, df=max(n - p, 1)))
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    # F-statistic for overall model significance
    if p > 1 and ss_tot > 0:
        ss_model = ss_tot - ss_res
        df_model = p - 1
        df_resid = n - p
        f_stat = (ss_model / df_model) / (ss_res / df_resid) if df_resid > 0 else np.nan
        f_pval = float(_ss.f.sf(f_stat, df_model, df_resid)) if np.isfinite(f_stat) else np.nan
    else:
        f_stat = f_pval = np.nan

    return dict(
        beta=beta, se=se, se_hc3=se_hc3,
        t_vals=t_vals, p_vals=p_vals, p_vals_hc3=p_vals_hc3,
        ci_lower=ci_lower, ci_upper=ci_upper,
        r_sq=r_sq, adj_r_sq=adj_r_sq,
        n=n, p=p,
        residuals=residuals, y_hat=y_hat,
        f_stat=f_stat, f_pval=f_pval,
    )


def _kfold_ols_predict(X: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    """
    k-fold cross-validated OLS predictions (no sklearn required).

    Returns predicted y values for all n observations, where each prediction
    comes from a model trained on the held-out fold's complement.
    """
    n = len(y)
    indices = np.arange(n)
    fold_size = n // k
    y_hat_cv = np.full(n, np.nan)

    for fold in range(k):
        val_start = fold * fold_size
        val_end   = (fold + 1) * fold_size if fold < k - 1 else n
        val_idx   = indices[val_start:val_end]
        trn_idx   = np.concatenate([indices[:val_start], indices[val_end:]])

        X_trn = np.column_stack([np.ones(len(trn_idx)), X[trn_idx]])
        X_val = np.column_stack([np.ones(len(val_idx)),  X[val_idx]])
        y_trn = y[trn_idx]

        try:
            beta, _, _, _ = np.linalg.lstsq(X_trn, y_trn, rcond=None)
            y_hat_cv[val_idx] = X_val @ beta
        except np.linalg.LinAlgError:
            y_hat_cv[val_idx] = np.mean(y_trn)

    return y_hat_cv


def compute_all_correlations(
    df: pd.DataFrame,
    outcome_cols: List[str],
    factor_cols: List[str],
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Compute Pearson r and Spearman ρ for every factor × outcome combination.

    Args:
        df:           Master county DataFrame.
        outcome_cols: List of outcome column names (must exist in df).
        factor_cols:  List of factor column names (must exist in df).
        min_n:        Minimum valid-pair count; row omitted if below threshold.

    Returns:
        DataFrame with columns:
            Outcome, Factor, Pearson_r, Pearson_p, Spearman_r, Spearman_p, N
        Sorted by |Pearson_r| descending within each outcome.
    """
    rows = []
    for oc in outcome_cols:
        if oc not in df.columns:
            continue
        for fc in factor_cols:
            if fc not in df.columns:
                continue
            pair = df[[oc, fc]].dropna()
            n = len(pair)
            if n < min_n:
                continue
            x = pair[fc].values.astype(float)
            y = pair[oc].values.astype(float)

            pr, pp = _ss.pearsonr(x, y)
            sr, sp = _ss.spearmanr(x, y)

            rows.append({
                "Outcome":   _OUTCOME_LABEL.get(oc, oc),
                "Factor":    _FACTOR_LABEL.get(fc, fc),
                "_outcome_col": oc,
                "_factor_col":  fc,
                "Pearson r":    float(pr),
                "Pearson p":    float(pp),
                "Spearman ρ":   float(sr),
                "Spearman p":   float(sp),
                "N":            int(n),
                "_abs_r":       abs(float(pr)),
            })

    if not rows:
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .sort_values(["Outcome", "_abs_r"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return result


def run_rf_feature_importance(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols: List[str],
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Rank county factors by Random Forest feature importance.

    Requires scikit-learn. Falls back to |Pearson r| ranking if unavailable
    (flagged in the returned DataFrame's 'method' column).

    Missing feature values are imputed with column-wise medians before fitting.

    Args:
        df:           Master county DataFrame.
        outcome_col:  Target column name.
        feature_cols: Predictor column names.
        n_estimators: Number of RF trees.
        random_state: Seed for reproducibility.

    Returns:
        (importance_df, error_str) — error_str is None on success.
        importance_df columns: Rank, Feature, Importance, Method
    """
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return None, "Fewer than 2 feature columns found in dataset."

    sub = df[[outcome_col] + available].dropna(subset=[outcome_col]).copy()
    n = len(sub)
    if n < 30:
        return None, f"Only {n} counties have valid outcome data — need ≥ 30."

    X_raw = sub[available].values.astype(float)
    y     = sub[outcome_col].values.astype(float)
    X     = _impute_median(X_raw)

    method = "Random Forest (sklearn)"
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state,
            n_jobs=-1, oob_score=True,
        )
        rf.fit(X, y)
        importances = rf.feature_importances_
    except ImportError:
        # Fallback: absolute Pearson r as importance proxy
        method = "|Pearson r| (scikit-learn not installed)"
        importances = np.array([
            abs(float(_ss.pearsonr(X[:, j], y)[0]))
            for j in range(X.shape[1])
        ])

    importance_df = pd.DataFrame({
        "Feature":    [_FACTOR_LABEL.get(c, c) for c in available],
        "_col":       available,
        "Importance": importances,
        "Method":     method,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df.insert(0, "Rank", range(1, len(importance_df) + 1))

    return importance_df, None


def run_ols_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: List[str],
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Fit an OLS regression model using numpy (no statsmodels dependency).

    Missing values are dropped list-wise (complete cases only).

    Args:
        df:             Master county DataFrame.
        outcome_col:    Target column name.
        predictor_cols: Predictor column names.

    Returns:
        (results_dict, error_str)

        results_dict keys:
            summary_df  — DataFrame: Variable, Coefficient, Std Error,
                          t-stat, p-value, Robust SE (HC3), Robust p,
                          CI Lower, CI Upper
            r_sq, adj_r_sq, n, f_stat, f_pval
            outcome_col, predictor_cols
    """
    available = [c for c in predictor_cols if c in df.columns]
    if not available:
        return None, "None of the requested predictor columns exist in the dataset."
    if outcome_col not in df.columns:
        return None, f"Outcome column '{outcome_col}' not found."

    sub = df[[outcome_col] + available].dropna()
    n = len(sub)
    p = len(available) + 1  # predictors + intercept

    if n < p + 5:
        return None, (
            f"Only {n} complete cases for {len(available)} predictors. "
            f"Need at least {p + 5}."
        )

    X_raw = sub[available].values.astype(float)
    y     = sub[outcome_col].values.astype(float)
    X_c   = np.column_stack([np.ones(n), X_raw])   # design matrix with constant

    ols = _ols_fit(X_c, y)

    param_names = ["(Intercept)"] + [_FACTOR_LABEL.get(c, c) for c in available]
    summary_df = pd.DataFrame({
        "Variable":        param_names,
        "Coefficient":     ols["beta"],
        "Std Error":       ols["se"],
        "t-stat":          ols["t_vals"],
        "p-value":         ols["p_vals"],
        "Robust SE (HC3)": ols["se_hc3"],
        "Robust p":        ols["p_vals_hc3"],
        "CI Lower":        ols["ci_lower"],
        "CI Upper":        ols["ci_upper"],
    })

    return {
        "summary_df":    summary_df,
        "r_sq":          ols["r_sq"],
        "adj_r_sq":      ols["adj_r_sq"],
        "n":             ols["n"],
        "f_stat":        ols["f_stat"],
        "f_pval":        ols["f_pval"],
        "outcome_col":   outcome_col,
        "predictor_cols": available,
    }, None


def compute_vif(df: pd.DataFrame, predictor_cols: List[str]) -> pd.DataFrame:
    """
    Variance Inflation Factors for a predictor set.

    VIF_j = 1 / (1 − R²_j), where R²_j comes from regressing predictor j on
    all other predictors (complete cases, intercept included). VIF > 5
    indicates problematic multicollinearity; VIF > 10 means the coefficient
    for that predictor is unreliable in a joint model.

    Returns DataFrame with columns: Variable, VIF (sorted descending).
    Empty DataFrame if fewer than 2 predictors have usable data.
    """
    available = [c for c in predictor_cols if c in df.columns]
    sub = df[available].dropna()
    if len(available) < 2 or len(sub) < len(available) + 5:
        return pd.DataFrame()

    X = sub.values.astype(float)
    rows = []
    for j, col in enumerate(available):
        others = np.delete(X, j, axis=1)
        design = np.column_stack([np.ones(len(X)), others])
        target = X[:, j]
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ beta
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / max(1.0 - r_sq, 1e-10)
        rows.append({"Variable": _FACTOR_LABEL.get(col, col), "VIF": vif})

    return (
        pd.DataFrame(rows)
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def run_rf_partial_dependence(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols: List[str],
    top_k: int = 3,
    grid_points: int = 20,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    One-way partial dependence curves for the top RF features.

    Fits a Random Forest, ranks features by importance, then for each of the
    top_k features sweeps a percentile grid (5th–95th) while holding all other
    features at their observed values, averaging predictions at each grid
    point. Requires scikit-learn (no fallback — PD is meaningless for the
    |Pearson r| proxy).

    Returns:
        ({feature_label: DataFrame[grid_value, avg_prediction]}, error_str)
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None, "Partial dependence requires scikit-learn."

    available = [c for c in feature_cols if c in df.columns]
    sub = df[[outcome_col] + available].dropna(subset=[outcome_col])
    if len(sub) < 50:
        return None, f"Only {len(sub)} counties with outcome data — need ≥ 50."

    X = _impute_median(sub[available].values.astype(float))
    y = sub[outcome_col].values.astype(float)

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               random_state=random_state, n_jobs=-1)
    rf.fit(X, y)

    order = np.argsort(rf.feature_importances_)[::-1][:top_k]
    curves = {}
    for j in order:
        grid = np.percentile(X[:, j], np.linspace(5, 95, grid_points))
        avg_pred = []
        X_mod = X.copy()
        for g in grid:
            X_mod[:, j] = g
            avg_pred.append(float(rf.predict(X_mod).mean()))
        label = _FACTOR_LABEL.get(available[j], available[j])
        curves[label] = pd.DataFrame({"grid_value": grid, "avg_prediction": avg_pred})

    return curves, None


def _kmeans_numpy(X: np.ndarray, k: int, n_iter: int = 100,
                  random_state: int = 42) -> np.ndarray:
    """Lloyd's algorithm fallback when scikit-learn is unavailable."""
    rng = np.random.default_rng(random_state)
    centers = X[rng.choice(len(X), size=k, replace=False)]
    labels = np.zeros(len(X), dtype=int)
    for it in range(n_iter):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
    return labels


def compute_county_clusters(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int = 4,
    random_state: int = 42,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Cluster counties into structural archetypes with K-means.

    Features are z-score standardized (population log-transformed if present)
    before clustering, so no single scale dominates. Clustering uses
    *structural* county characteristics — outcomes are deliberately excluded
    so that comparing COVID outcomes across archetypes remains meaningful.

    Uses scikit-learn KMeans when available; a numpy Lloyd's-algorithm
    fallback otherwise (same random_state for reproducibility).

    Returns:
        (assignments_df, profile_df, error_str)
        assignments_df: countyFIPS, County Name, State, cluster (int)
        profile_df:     one row per cluster — county count plus the mean of
                        each feature and of available outcome columns.
    """
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 3:
        return None, None, "Fewer than 3 clustering features available."

    id_cols = [c for c in ["countyFIPS", "County Name", "State"] if c in df.columns]
    outcome_cols = [c for c in ["cases_per_100k", "deaths_per_100k",
                                "case_fatality_rate", "vax_complete_pct"]
                    if c in df.columns]

    sub = df[id_cols + available + outcome_cols].dropna(subset=available).copy()
    if len(sub) < k * 10:
        return None, None, f"Only {len(sub)} complete-case counties — need ≥ {k * 10}."

    X = sub[available].astype(float).copy()
    if "population" in X.columns:
        X["population"] = np.log10(X["population"].clip(lower=1))
    X = (X - X.mean()) / X.std().replace(0, 1)
    X_arr = X.values

    try:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=random_state,
                        n_init=10).fit_predict(X_arr)
    except ImportError:
        labels = _kmeans_numpy(X_arr, k, random_state=random_state)

    sub["cluster"] = labels.astype(int)

    profile = (
        sub.groupby("cluster")
        .agg(counties=("cluster", "size"),
             **{c: (c, "mean") for c in available + outcome_cols})
        .reset_index()
    )

    return sub[id_cols + ["cluster"]].reset_index(drop=True), profile, None


def compute_resilience_scores(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols: List[str],
    cv_folds: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Compute county resilience scores via cross-validated model predictions.

    Resilience Score = predicted outcome − actual outcome.
    Positive score → county performed better than expected given its
    structural characteristics (fewer deaths/cases than the model predicts).
    Negative score → county performed worse than expected.

    Method (in order of preference):
        1. scikit-learn RandomForestRegressor + cross_val_predict (captures
           non-linear factor interactions)
        2. k-fold cross-validated OLS (numpy fallback; linear relationships only)

    Cross-validation prevents data leakage: each county's prediction comes
    from a model that did NOT train on that county.

    Args:
        df:           Master county DataFrame.
        outcome_col:  Column to predict (typically "deaths_per_100k").
        feature_cols: AHRF predictor columns.
        cv_folds:     Number of cross-validation folds.
        n_estimators: RF trees (ignored for OLS fallback).
        random_state: Seed.

    Returns:
        (scores_df, error_str)

        scores_df columns:
            countyFIPS, County Name, State,
            actual, predicted, resilience_score, method
    """
    id_cols   = [c for c in ["countyFIPS", "County Name", "State"] if c in df.columns]
    available = [c for c in feature_cols if c in df.columns]

    if len(available) < 2:
        return None, "Fewer than 2 feature columns found."
    if outcome_col not in df.columns:
        return None, f"Outcome column '{outcome_col}' not found."

    keep = id_cols + [outcome_col] + available
    sub  = df[[c for c in keep if c in df.columns]].dropna(subset=[outcome_col]).copy()
    n    = len(sub)
    if n < max(cv_folds * 10, 50):
        return None, f"Only {n} counties with valid outcome data — need ≥ {max(cv_folds*10, 50)}."

    X_raw = sub[available].values.astype(float)
    y     = sub[outcome_col].values.astype(float)
    X     = _impute_median(X_raw)

    method = "Unknown"
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_predict

        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
        )
        predicted = cross_val_predict(rf, X, y, cv=cv_folds)
        method = f"Random Forest ({cv_folds}-fold CV)"

    except ImportError:
        predicted = _kfold_ols_predict(X, y, k=cv_folds)
        method = f"OLS ({cv_folds}-fold CV, scikit-learn not installed)"

    result = sub[id_cols + [outcome_col]].copy().reset_index(drop=True)
    result["actual"]           = y
    result["predicted"]        = predicted
    result["resilience_score"] = predicted - y     # positive = better than expected
    result["method"]           = method
    result = result.drop(columns=[outcome_col], errors="ignore")

    return result, None


def generate_ols_interpretation(
    summary_df: pd.DataFrame,
    outcome_label: str,
    r_sq: float,
    adj_r_sq: float,
    n: int,
) -> List[str]:
    """
    Generate rule-based interpretation bullets from an OLS results table.

    Returns a list of markdown strings suitable for st.markdown() bullets.
    Does not use AI text generation — purely rule-based from model statistics.
    """
    bullets = []

    # Overall model fit
    fit_word = (
        "strong" if r_sq >= 0.5 else
        "moderate" if r_sq >= 0.25 else
        "weak"
    )
    bullets.append(
        f"The model explains **{r_sq * 100:.1f}%** of the variation in {outcome_label} "
        f"(R² = {r_sq:.3f}, adj. R² = {adj_r_sq:.3f}, N = {n:,}) — a **{fit_word}** overall fit."
    )

    # Significant predictors
    sig = summary_df[
        (summary_df["Variable"] != "(Intercept)") &
        (summary_df["p-value"] < 0.05)
    ].copy()
    non_sig = summary_df[
        (summary_df["Variable"] != "(Intercept)") &
        (summary_df["p-value"] >= 0.05)
    ]

    if sig.empty:
        bullets.append(
            "No individual predictor reached statistical significance (p < 0.05) "
            "after accounting for other variables in the model."
        )
    else:
        for _, row in sig.iterrows():
            direction = "positively" if row["Coefficient"] > 0 else "negatively"
            bullets.append(
                f"**{row['Variable']}** is {direction} associated with {outcome_label} "
                f"after controlling for other predictors "
                f"(β = {row['Coefficient']:.3f}, p = {row['p-value']:.4f})."
            )

    if not non_sig.empty:
        non_sig_names = ", ".join(f"**{v}**" for v in non_sig["Variable"])
        bullets.append(
            f"The following predictors were not statistically significant at p < 0.05: "
            f"{non_sig_names}."
        )

    bullets.append(
        "_These results show statistical associations, not causal relationships. "
        "Confounding variables, measurement error, and ecological fallacy apply._"
    )

    return bullets
