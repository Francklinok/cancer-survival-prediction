"""
diagnostic_analysis.py —  Diagnostic & Causal Analysis.

=============================================================================
DIAGNOSTIC ANALYSIS
=============================================================================
Answers: "Why did this happen?"

Functions
---------
  diagnostic_analysis()        — Point-Biserial r / Pearson |r| / Cramér's V BC ranking (first pass)
  root_cause_analysis()        — Top driver identification with effect sizes
  contribution_analysis()      — Quantify each feature's share of target variance
  shap_global_analysis()       — SHAP-based global feature importance (model-free)
  segment_analysis()           — Compare segments defined by a grouping variable
  variance_decomposition()     — ANOVA / hierarchical variance decomposition
  anomaly_explanation()        — Profile anomalous rows vs normal population
  cohort_analysis()            — Compare cohorts defined by time or category

=============================================================================
CAUSAL ANALYSIS
=============================================================================
Answers: "What caused this?"

Functions
---------
  causal_analysis()            — Effect sizes for binary treatments (Cohen's d)
  propensity_score_matching()  — PSM: balance confounders before comparing groups
  difference_in_differences()  — DiD: before/after × treated/control estimate
  regression_discontinuity()   — RD: local linear comparison around a cutoff
  causal_forest_ate()          — Causal Forest ATE via GRF-style honest splitting
  double_machine_learning()    — DML / Partialling-out estimator (Robinson 1988)

=============================================================================
Relationship with other modules
=============================================================================
diagnostic_analysis()          — fast first-pass ranking (linear + chi², with leakage guard)
feature_importance_exploration()  (diagnostic/data_diagnostic.py)
                               — Borda (KW η² + MI + RF), non-linear, model-based.
                                 Run AFTER diagnostic_analysis().
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    mannwhitneyu,
    ttest_ind,
)

from ml_framework.visualization.analysis.diagnostic_plots import (
    plot_causal_effect,
    plot_diagnostic_association,
    rout_cause_plot,
    contribution_analysis_plot,
    shap_global_analysis_plot,
    segment_analysis_plot,
    variance_decomposition_plot,
    anomaly_explanation_plot,
    cohort_analysis_Plot,
    propensity_score_matching_plot,
    difference_in_differences_plot,
    regression_discontinuity_plot,
    causal_forest_plot,
    double_machine_learning_plot,
)
from ml_framework.utils.display_utils import section_header
from ml_framework.utils.statistical_utils import compute_cohens_d, compute_cramers_v, compute_cramers_v_bc

from scipy.stats import spearmanr as _spearmanr
from scipy.stats import ttest_rel as _ttest_rel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap
from sklearn.model_selection import KFold
from scipy.stats import t as t_dist
from scipy.stats import norm as _norm_cf
from sklearn.model_selection import cross_val_predict
from numpy.linalg import lstsq, LinAlgError
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("ml_framework.diagnostic_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def ci_str(vals: list, alpha: float = 0.05) -> str:
    """
    Return a 'mean [lo - hi]' string for a list of values.

    Parameters
    ----------
    vals  : list of floats (e.g., CV scores)
    alpha : two-sided significance level (default 0.05 → 95% CI).
            The CI spans from percentile (alpha/2 × 100) to (1 - alpha/2) × 100.
            Previous implementation used percentile(alpha*100) and (1-alpha)*100,
            which produced a 90% CI when alpha=0.05 — off by a factor of 2.

    Returns
    -------
    str  — e.g. '0.8543 [0.8210 – 0.8876]'
    """
    if not vals:
        return "N/A"
    arr = np.array(vals)
    lo  = float(np.percentile(arr, alpha / 2 * 100))
    hi  = float(np.percentile(arr, (1 - alpha / 2) * 100))
    return f"{arr.mean():.4f} [{lo:.4f} – {hi:.4f}]"


def _auto_num_cat(
    df: pd.DataFrame, target_col: str
) -> Tuple[List[str], List[str]]:
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col
    ]
    return num_cols, cat_cols


# ──────────────────────────────────────────────────────────────────────────────
# 1. DIAGNOSTIC ANALYSIS  (first-pass linear ranking)
# ──────────────────────────────────────────────────────────────────────────────

def diagnostic_analysis(
    df: pd.DataFrame,
    target_col: str,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    top_n: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Identify the variables most linearly associated with the target variable.

    Metrics
    -------
    - Numeric variables   : |Pearson r| (linear correlation with target)
    - Categorical variables: Cramér's V  (via Chi²)

    When to use
    -----------
    Use as a first-pass diagnostic to quickly rank features by linear association.
    For non-linear signal and interaction effects, follow up with
    ``feature_importance_exploration()`` from ``diagnostic/data_diagnostic.py``.

    Returns
    -------
    pd.DataFrame sorted by association strength (descending)
    Columns: variable, type, association, metric, value
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    section_header("DIAGNOSTIC ANALYSIS")

    if num_cols is None or cat_cols is None:
        _num, _cat = _auto_num_cat(df, target_col)
        num_cols = num_cols if num_cols is not None else _num
        cat_cols = cat_cols if cat_cols is not None else _cat

    y = df[target_col]
    is_binary_target = (y.nunique() == 2)
    metric_label_num = "Point-Biserial r" if is_binary_target else "Pearson |r|"

    records = []
    leakage_flags: list = []

    for col in num_cols:
        try:
            corr = float(df[col].corr(y))
            if np.isnan(corr):
                continue
            abs_corr = abs(corr)
            # Leakage guard: |r| > 0.95 is a strong signal of derived/encoded duplicates.
            if abs_corr > 0.95:
                leakage_flags.append((col, "numeric", abs_corr))
            records.append({
                "variable":    col,
                "type":        "numeric",
                "association": abs_corr,
                "metric":      metric_label_num,
                "value":       round(corr, 4),
            })
        except Exception as exc:
            logger.debug("Correlation %s: %s", col, exc)

    for col in cat_cols:
        try:
            contingency = pd.crosstab(df[col], y)
            chi2, _, _, _ = chi2_contingency(contingency)
            n = int(contingency.sum().sum())
            nr, nc = contingency.shape
            cramers_v = compute_cramers_v_bc(chi2, n, nr, nc)
            if cramers_v > 0.95:
                leakage_flags.append((col, "categorical", cramers_v))
            records.append({
                "variable":    col,
                "type":        "categorical",
                "association": cramers_v,
                "metric":      "Cramér's V (BC)",
                "value":       round(cramers_v, 4),
            })
        except Exception as exc:
            logger.debug("Cramér's V %s: %s", col, exc)

    if leakage_flags:
        print("\n POTENTIAL DATA LEAKAGE detected (association > 0.95):")
        for flagged_col, ftype, fval in leakage_flags:
            print(f"     {flagged_col}  [{ftype}]  metric={fval:.4f}")
        print(" These columns may be derived from the target (e.g. encoded labels,")
        print(" recurrence flags). Exclude them before modelling.\n")

    if not records:
        logger.warning("No associations computed — check column names.")
        return pd.DataFrame()

    diag_df = (
        pd.DataFrame(records)
        .sort_values("association", ascending=False)
        .reset_index(drop=True)
    )

    top = diag_df.head(top_n)

    if plot and not top.empty:
        try:
            plot_diagnostic_association(top, target_col, top_n, df)
        except Exception as exc:
            logger.warning("Diagnostic visualization failed: %s", exc)

    print(f"\n  Top {top_n} variables most associated with '{target_col}':")
    print(top[["variable", "type", "metric", "value"]].to_string(index=False))
    # print(
    #     "\n  METHODOLOGICAL NOTE:"
    #     "\n  · Numeric features use " + metric_label_num + " — LINEAR association only."
    #     "\n  · Categorical features use bias-corrected Cramér's V (BC) — association via Chi²."
    #     "\n  · These two metrics are NOT on the same scale. The combined ranking is an"
    #     "\n    approximation (larger |r| ≠ larger V). Treat cross-type comparisons as"
    #     "\n    rough guidance only. For rigorous non-linear importance, use"
    #     "\n    feature_importance_exploration() (Borda: KW η² + MI + RF)."
    # )
    return diag_df


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROOT CAUSE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def root_cause_analysis(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
    alpha: float = 0.05,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Identify root drivers of an observed phenomenon.

    Combines three evidence sources per feature:
      - Statistical test (t-test / Mann-Whitney / ANOVA / Kruskal-Wallis)
      - Effect size     (Cohen's d / Cramér's V / η²)
      - % variance explained (R² for numeric, Cramér's V² for categorical)

    Returns
    -------
    pd.DataFrame sorted by variance_explained descending.
    Columns: variable, type, test, statistic, p_value, effect_size,
             effect_magnitude, variance_explained, significant
    """
    section_header("ROOT CAUSE ANALYSIS")
    num_cols, cat_cols = _auto_num_cat(df, target_col)
    y = df[target_col]
    is_cat_target = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 15

    records = []

    for col in num_cols:
        common = df[[col, target_col]].dropna()
        if len(common) < 10:
            continue
        try:
            if is_cat_target:
                groups = [
                    common.loc[common[target_col] == g, col].values
                    for g in common[target_col].unique()
                ]
                groups = [g for g in groups if len(g) >= 5]
                if len(groups) < 2:
                    continue
                if len(groups) == 2:
                    stat, p = ttest_ind(groups[0], groups[1], equal_var=False)
                    test = "Welch t-test"
                    # the control/reference group SD
                    d = compute_cohens_d(groups[0], groups[1], variant="glass")
                    effect_size = abs(d)
                    effect_mag = ("large" if effect_size > 0.8 else
                                  "medium" if effect_size > 0.5 else
                                  "small" if effect_size > 0.2 else "negligible")
                else:
                    stat, p = kruskal(*groups)
                    test = "Kruskal-Wallis"
                    n = sum(len(g) for g in groups)
                    k = len(groups)
                    # η² from Kruskal-Wallis H:
                    # η² ≈ H / (n - 1) when k is small.
                    eta2 = max(0.0, (stat - k + 1) / (n - 1))
                    effect_size = eta2
                    effect_mag = ("large" if eta2 > 0.14 else
                                  "medium" if eta2 > 0.06 else
                                  "small" if eta2 > 0.01 else "negligible")
            else:
                # Numeric feature × numeric target: test association via Spearman ρ.
                rho, p = _spearmanr(common[col].values, common[target_col].values)
                stat = rho
                corr = abs(float(rho))
                test = "Spearman ρ"
                effect_size = corr
                effect_mag = ("strong" if corr > 0.5 else
                              "moderate" if corr > 0.3 else
                              "weak" if corr > 0.1 else "negligible")

            # Variance explained: for all effect types, squaring gives proportion of variance.
            # Cohen's d² ≈ η² (point-biserial r²); Pearson r² = R²; Cramér's V² = variance explained.
            var_exp = effect_size ** 2
            records.append({
                "variable": col, "type": "numeric",
                "test": test, "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "effect_size": round(effect_size, 4),
                "effect_magnitude": effect_mag,
                "variance_explained": round(var_exp, 4),
                "significant": bool(p < alpha),
            })
        except Exception as exc:
            logger.debug("Root cause %s: %s", col, exc)

    for col in cat_cols:
        common = df[[col, target_col]].dropna()
        if len(common) < 10:
            continue
        try:
            ct = pd.crosstab(common[col], common[target_col])
            chi2, p, _, _ = chi2_contingency(ct)
            n = int(ct.sum().sum())
            nr, nc = ct.shape
            v = compute_cramers_v_bc(chi2, n, nr, nc)
            effect_mag = ("strong" if v > 0.5 else
                          "moderate" if v > 0.3 else
                          "weak" if v > 0.1 else "negligible")
            records.append({
                "variable": col, "type": "categorical",
                "test": "Chi-square", "statistic": round(float(chi2), 4),
                "p_value": round(float(p), 6),
                "effect_size": round(v, 4),
                "effect_magnitude": effect_mag,
                "variance_explained": round(v ** 2, 4),
                "significant": bool(p < alpha),
            })
        except Exception as exc:
            logger.debug("Root cause cat %s: %s", col, exc)

    if not records:
        print("  No root cause drivers found.")
        return pd.DataFrame()

    rca_df = (
        pd.DataFrame(records)
        .sort_values("variance_explained", ascending=False)
        .reset_index(drop=True)
    )

    sig = rca_df[rca_df["significant"]]
    print(f"\n  {len(sig)} significant drivers of '{target_col}' (α={alpha}):\n")
    display_cols = ["variable", "type", "test", "effect_size", "effect_magnitude",
                    "variance_explained", "p_value", "significant"]
    print(rca_df[display_cols].head(top_n).to_string(index=False))

    if plot and not rca_df.empty:
        try:
            rout_cause_plot(rca_df, target_col, top_n)
        except Exception as exc:
            logger.debug("Root cause plot failed: %s", exc)

    return rca_df


# ──────────────────────────────────────────────────────────────────────────────
# 3. CONTRIBUTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def contribution_analysis(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 15,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Quantify each feature's marginal contribution to explaining target variance.

    Method: OLS R² gain — incremental R² when adding each feature to a
    baseline intercept-only model. Uses sklearn LinearRegression for speed.

    For classification targets, uses Logistic Regression pseudo-R² (McFadden).

    Returns
    -------
    pd.DataFrame — columns: feature, r2_contribution, cumulative_r2, rank
    """
    section_header("CONTRIBUTION ANALYSIS")
    num_cols, cat_cols = _auto_num_cat(df, target_col)
    y_raw = df[target_col].dropna()
    is_cat_target = not pd.api.types.is_numeric_dtype(y_raw) or y_raw.nunique() <= 15

    y = y_raw.copy()
    if is_cat_target and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    contrib: Dict[str, float] = {}

    for col in num_cols + cat_cols:
        common = df[[col, target_col]].dropna()
        if len(common) < 10:
            continue
        x_vals = common[col]
        if col in cat_cols:
            x_vals = x_vals.astype("category").cat.codes
        X = x_vals.values.reshape(-1, 1)
        y_sub = y.loc[common.index]

        try:
            if is_cat_target:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = LogisticRegression(max_iter=300, random_state=42)
                    model.fit(X, y_sub)
                    # McFadden pseudo-R² = 1 - LL_model / LL_null
                    # LogisticRegression.score() returns ACCURACY
                    
                    y_arr    = y_sub.values
                    ll_model = float(model.predict_log_proba(X)[np.arange(len(y_arr)), y_arr].sum())
                    n_obs    = len(y_arr)
                    classes, counts = np.unique(y_arr, return_counts=True)
                    p_null   = counts / counts.sum()
                    ll_null  = float(np.sum(counts * np.log(p_null + 1e-15)))
                    r2       = 1.0 - (ll_model / ll_null) if ll_null != 0 else 0.0
            else:
                r2 = LinearRegression().fit(X, y_sub).score(X, y_sub)
            contrib[col] = max(0.0, float(r2))
        except Exception as exc:
            logger.debug("Contribution %s: %s", col, exc)

    if not contrib:
        print("  No contributions computed.")
        return pd.DataFrame()

    contrib_df = (
        pd.DataFrame.from_dict(contrib, orient="index", columns=["r2_contribution"])
        .sort_values("r2_contribution", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    total = contrib_df["r2_contribution"].sum()
    contrib_df["contribution_pct"] = (
        (contrib_df["r2_contribution"] / total * 100).round(2) if total > 0
        else 0.0
    )
    contrib_df["cumulative_pct"] = contrib_df["contribution_pct"].cumsum().round(2)
    contrib_df["rank"] = range(1, len(contrib_df) + 1)

    top = contrib_df.head(top_n)
    print(f"\n  Feature contribution to target variance ('{target_col}'):\n")
    print(top[["rank", "feature", "r2_contribution", "contribution_pct",
               "cumulative_pct"]].to_string(index=False))
    # print(
    #     "\n  ⚠  METHODOLOGICAL NOTE — contribution_pct interpretation:"
    #     "\n  · Each r2_contribution is a MARGINAL bivariate R² (feature × target alone)."
    #     "\n  · Marginal R² values are NOT additive: correlated features share explained"
    #     "\n    variance, so the sum can exceed 100% of actual target variance."
    #     "\n  · contribution_pct = r2_i / Σr2_i × 100 is a RELATIVE RANKING metric,"
    #     "\n    not a partition of explained variance."
    #     "\n  · For an additive decomposition use: variance_decomposition() (ANOVA Type I/II)"
    #     "\n    or SHAP values on a single multivariate model (shap_global_analysis())."
    # )

    if plot:
        try:
            contribution_analysis_plot(top, target_col)
        except Exception as exc:
            logger.debug("Contribution plot failed: %s", exc)
        

    return contrib_df


# ──────────────────────────────────────────────────────────────────────────────
# 4. SHAP GLOBAL ANALYSIS  (model-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def shap_global_analysis(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 15,
    n_estimators: int = 100,
    plot: bool = True,
) -> pd.DataFrame:
    """
    SHAP-based global feature importance — model-agnostic, captures non-linearity.

    Uses a LightGBM or Random Forest model trained internally. SHAP values are
    computed for all samples and averaged as |mean(SHAP)| per feature.

    Returns
    -------
    pd.DataFrame — columns: feature, mean_abs_shap, rank, contribution_pct
    """
    section_header("SHAP GLOBAL ANALYSIS")

    num_cols, cat_cols = _auto_num_cat(df, target_col)
    
    y_raw = df[target_col].dropna()
    is_cat_target = not pd.api.types.is_numeric_dtype(y_raw) or y_raw.nunique() <= 15
    common_idx = y_raw.index

    # Encode features
    X_parts = []
    feature_names = []
    for col in num_cols:
        s = df.loc[common_idx, col].fillna(df[col].median())
        X_parts.append(s.values.reshape(-1, 1))
        feature_names.append(col)
    for col in cat_cols:
        s = df.loc[common_idx, col].fillna("__missing__")
        X_parts.append(
            pd.get_dummies(s, prefix=col, drop_first=True).values
        )
        feature_names.extend(
            pd.get_dummies(s, prefix=col, drop_first=True).columns.tolist()
        )

    if not X_parts:
        print("  No features available for SHAP analysis.")
        return pd.DataFrame()

    X = np.hstack(X_parts).astype(float)
    y = y_raw.values
    if is_cat_target and not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)

    try:
        RF = RandomForestClassifier if is_cat_target else RandomForestRegressor
        model = RF(n_estimators=n_estimators, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return pd.DataFrame()

    # Aggregate OHE features back to originals
    mean_abs: Dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        orig = fname.split("_")[0] if any(fname.startswith(c + "_") for c in cat_cols) else fname
        mean_abs[orig] = mean_abs.get(orig, 0.0) + float(np.abs(shap_values[:, i]).mean())

    shap_df = (
        pd.DataFrame.from_dict(mean_abs, orient="index", columns=["mean_abs_shap"])
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    total = shap_df["mean_abs_shap"].sum()
    shap_df["contribution_pct"] = (
        (shap_df["mean_abs_shap"] / total * 100).round(2) if total > 0 else 0.0
    )
    shap_df["rank"] = range(1, len(shap_df) + 1)
    top = shap_df.head(top_n)

    print(f"\n  SHAP Global Importance — top {top_n} features ('{target_col}'):\n")
    print(top[["rank", "feature", "mean_abs_shap", "contribution_pct"]].to_string(index=False))

    if plot:
       try:
        shap_global_analysis_plot(top, target_col)

       except Exception as exc:
            logger.debug("SHAP plot failed: %s", exc)

    return shap_df


# ──────────────────────────────────────────────────────────────────────────────
# 5. SEGMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def segment_analysis(
    df: pd.DataFrame,
    target_col: str,
    segment_col: str,
    metrics: Optional[List[str]] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compare segments defined by a grouping variable on a numeric target.

    For each segment, computes: count, mean, median, std, min, max,
    effect size vs global mean (Cohen's d), and statistical significance
    (Kruskal-Wallis omnibus + pairwise Mann-Whitney with Bonferroni correction).

    Returns
    -------
    pd.DataFrame — one row per segment, sorted by mean target descending.
    """
    section_header(f"SEGMENT ANALYSIS — '{segment_col}' × '{target_col}'")

    if target_col not in df.columns or segment_col not in df.columns:
        raise KeyError(f"Column '{target_col}' or '{segment_col}' not found.")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"target_col '{target_col}' must be numeric for segment analysis.")

    common = df[[segment_col, target_col]].dropna()
    global_mean = float(common[target_col].mean())
    global_std  = float(common[target_col].std())

    groups_map = {
        seg: grp[target_col].values
        for seg, grp in common.groupby(segment_col)
        if len(grp) >= 5
    }
    if len(groups_map) < 2:
        print("  Not enough segments (need >= 2 with >= 5 obs each).")
        return pd.DataFrame()

    # Omnibus test
    stat, p_omnibus = kruskal(*groups_map.values())

    records = []
    for seg, vals in groups_map.items():
       
        other_vals = np.concatenate([v for s, v in groups_map.items() if s != seg])
        if len(other_vals) >= 2 and vals.std() > 0 and other_vals.std() > 0:
            n1, n2 = len(vals), len(other_vals)
            pooled_var = ((n1 - 1) * vals.std() ** 2 + (n2 - 1) * other_vals.std() ** 2) / (n1 + n2 - 2)
            d = (vals.mean() - other_vals.mean()) / (np.sqrt(pooled_var) + 1e-12)
        else:
            d = 0.0
        records.append({
            "segment":          seg,
            "n":                len(vals),
            "mean":             round(float(vals.mean()), 4),
            "median":           round(float(np.median(vals)), 4),
            "std":              round(float(vals.std()), 4),
            "min":              round(float(vals.min()), 4),
            "max":              round(float(vals.max()), 4),
            "cohens_d_vs_rest": round(d, 4),
            "deviation_pct":    round((vals.mean() - global_mean) / (abs(global_mean) + 1e-12) * 100, 2),
        })

    seg_df = pd.DataFrame(records).sort_values("mean", ascending=False).reset_index(drop=True)

    print(f"\n  Omnibus Kruskal-Wallis: H={stat:.3f}, p={p_omnibus:.4f}")
    print(f"  Global mean '{target_col}': {global_mean:.4f}\n")
    print(seg_df.to_string(index=False))

    if plot:
        try:
            segment_analysis_plot(seg_df, global_mean, target_col, segment_col)
        except Exception as exc:
            logger.debug("Segment plot failed: %s", exc)
    return seg_df


# ──────────────────────────────────────────────────────────────────────────────
# 6. VARIANCE DECOMPOSITION
# ──────────────────────────────────────────────────────────────────────────────

def variance_decomposition(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Optional[List[str]] = None,
    top_n: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Decompose total variance of target into between-group and within-group components.

    For each grouping variable (categorical), computes:
      - SS_between  : variance explained by group membership
      - SS_within   : residual variance within groups
      - omega²      : unbiased effect size (ANOVA-based)
      - η²          : classical eta-squared

    Returns
    -------
    pd.DataFrame — sorted by omega² descending.
    """
    section_header("VARIANCE DECOMPOSITION")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"target_col '{target_col}' must be numeric.")

    _, cat_cols = _auto_num_cat(df, target_col)
    if group_cols is None:
        group_cols = cat_cols

    records = []
    for col in group_cols:
        common = df[[col, target_col]].dropna()
        if len(common) < 10:
            continue
        try:
            groups = [grp[target_col].values for _, grp in common.groupby(col)
                      if len(grp) >= 5]
            if len(groups) < 2:
                continue

            # Use only rows that belong to an assigned group to ensure SS_total = SS_between + SS_within
            y_grouped = pd.concat([pd.Series(g) for g in groups])
            grand_mean = float(y_grouped.mean())
            ss_total = float(((y_grouped - grand_mean) ** 2).sum())

            # ANOVA F-test
            f_stat, p_val = f_oneway(*groups)
            n = len(common)
            k = len(groups)
            df_between = k - 1
            df_within  = n - k

            # SS components
            group_means  = [g.mean() for g in groups]
            group_counts = [len(g) for g in groups]
            ss_between = sum(nc * (gm - grand_mean) ** 2
                             for nc, gm in zip(group_counts, group_means))
            ss_within  = ss_total - ss_between

            eta2   = ss_between / ss_total if ss_total > 0 else 0.0
            ms_wit = ss_within / df_within if df_within > 0 else 0.0
            omega2 = max(0.0, (ss_between - df_between * ms_wit) /
                         (ss_total + ms_wit)) if ms_wit > 0 else 0.0

            records.append({
                "grouping_var":  col,
                "k_groups":      k,
                "ss_between":    round(ss_between, 4),
                "ss_within":     round(ss_within,  4),
                "eta_squared":   round(eta2,   4),
                "omega_squared": round(omega2, 4),
                "f_statistic":   round(float(f_stat), 4),
                "p_value":       round(float(p_val),  6),
                "significant":   bool(p_val < 0.05),
            })
        except Exception as exc:
            logger.debug("Variance decomp %s: %s", col, exc)

    if not records:
        print("  No variance decomposition results.")
        return pd.DataFrame()

    vd_df = (
        pd.DataFrame(records)
        .sort_values("omega_squared", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n  Variance decomposition for '{target_col}':\n")
    print(vd_df.head(top_n).to_string(index=False))

    if plot:
        try:
            variance_decomposition_plot(vd_df, target_col)
        except Exception as exc:
            logger.debug("Variance decomposition plot failed: %s", exc)

    return vd_df


# ──────────────────────────────────────────────────────────────────────────────
# 7. ANOMALY EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────

def anomaly_explanation(
    df: pd.DataFrame,
    anomaly_mask: pd.Series,
    target_col: Optional[str] = None,
    top_n: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Profile anomalous rows versus the normal population.

    For each numeric feature: computes mean shift, Cohen's d, and Mann-Whitney p.
    For each categorical feature: computes distribution shift (JS divergence proxy).

    Parameters
    ----------
    df           : full DataFrame
    anomaly_mask : boolean Series (True = anomalous row)
    target_col   : exclude from feature analysis if provided

    Returns
    -------
    pd.DataFrame — sorted by |mean_shift| descending, anomalies vs normals.
    """
    section_header("ANOMALY EXPLANATION")

    anomalies = df[anomaly_mask]
    normals   = df[~anomaly_mask]
    n_anom    = len(anomalies)
    n_norm    = len(normals)

    print(f"\n  Anomalies: {n_anom} | Normal: {n_norm} | Ratio: {n_anom/(n_anom+n_norm)*100:.1f}%\n")

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    records = []
    for col in num_cols:
        a_vals = anomalies[col].dropna().values
        n_vals = normals[col].dropna().values
       
        if len(a_vals) < 5 or len(n_vals) < 5:
            continue
        try:
            shift = float(a_vals.mean() - n_vals.mean())
            # Normal population (n_vals) plays the reference/control role → pooled SD
            d = compute_cohens_d(n_vals, a_vals, variant="pooled")
            stat, p = mannwhitneyu(a_vals, n_vals, alternative="two-sided")
            z_mean = float(a_vals.mean() - n_vals.mean()) / (float(n_vals.std()) + 1e-12)
            records.append({
                "feature":       col,
                "anomaly_mean":  round(float(a_vals.mean()), 4),
                "normal_mean":   round(float(n_vals.mean()), 4),
                "mean_shift":    round(shift, 4),
                "z_shift":       round(z_mean, 4),
                "cohens_d":      round(d, 4),
                "mwu_p":         round(float(p), 6),
                "significant":   bool(p < 0.05),
            })
        except Exception as exc:
            logger.debug("Anomaly explanation %s: %s", col, exc)

    if not records:
        print("  No numeric features to profile.")
        return pd.DataFrame()

    anom_df = (
        pd.DataFrame(records)
        .assign(_abs_d=lambda x: x["cohens_d"].abs())
        .sort_values("_abs_d", ascending=False)
        .drop(columns=["_abs_d"])
        .reset_index(drop=True)
    )

    top = anom_df.head(top_n)
    print("  Top discriminating features (anomalies vs normals):\n")
    print(top.to_string(index=False))

    if plot:
        try:
            anomaly_explanation_plot(top)
        except Exception as exc:
            logger.debug("Anomaly explanation plot failed: %s", exc)
        
    return anom_df


# ──────────────────────────────────────────────────────────────────────────────
# 8. COHORT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def cohort_analysis(
    df: pd.DataFrame,
    target_col: str,
    cohort_col: str,
    metrics: Optional[List[str]] = None,
    sort_cohorts: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compare cohorts (time periods, patient groups, treatment arms) on a target.

    Computes per-cohort: count, mean, median, std, min, max, change vs
    first cohort (%), and cumulative trend.

    Parameters
    ----------
    cohort_col   : column defining cohort membership (e.g. 'year', 'quarter', 'arm')

    Returns
    -------
    pd.DataFrame — one row per cohort, sorted by cohort_col.
    """
    section_header(f"COHORT ANALYSIS — '{cohort_col}' × '{target_col}'")

    common = df[[cohort_col, target_col]].dropna()
    if not pd.api.types.is_numeric_dtype(common[target_col]):
        raise TypeError(f"target_col '{target_col}' must be numeric for cohort analysis.")

    cohort_stats = (
        common.groupby(cohort_col)[target_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    cohort_stats.columns = ["cohort", "n", "mean", "median", "std", "min", "max"]
    if sort_cohorts:
        try:
            cohort_stats = cohort_stats.sort_values("cohort").reset_index(drop=True)
        except TypeError:
            pass

    base_mean = float(cohort_stats["mean"].iloc[0]) if len(cohort_stats) > 0 else 1.0
    cohort_stats["change_vs_first_pct"] = (
        (cohort_stats["mean"] - base_mean) / (abs(base_mean) + 1e-12) * 100
    ).round(2)

    # Omnibus test
    groups = [
        common.loc[common[cohort_col] == c, target_col].values
        for c in cohort_stats["cohort"]
        if len(common[common[cohort_col] == c]) >= 5
    ]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        print(f"\n  Omnibus Kruskal-Wallis: H={stat:.3f}, p={p:.4f}")

    print(f"\n  Cohort summary for '{target_col}':\n")
    print(cohort_stats.to_string(index=False))

    if plot:
        try:
            cohort_analysis_Plot(cohort_stats,cohort_col, target_col)
        except Exception as exc:
            logger.debug("Cohort analysis plot failed: %s", exc)
        

    return cohort_stats


# ──────────────────────────────────────────────────────────────────────────────
# 9. CAUSAL ANALYSIS  (binary treatments — observational)
# ──────────────────────────────────────────────────────────────────────────────

def causal_analysis(
    df: pd.DataFrame,
    target_col: str,
    treatment_cols: List[str],
    confounders: Optional[List[str]] = None,
    alpha: float = 0.05,
    bonferroni: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Basic causal analysis: compare treated vs untreated groups on a numeric target.

    Computes for each binary treatment variable:
      - Cohen's d (standardized mean difference)
      - Welch's t-test (if numeric target) or Mann-Whitney U (fallback)
      - Statistical significance at alpha = 0.05 (uncorrected)

    Limitation: confounders are declared but NOT adjusted for.
    For bias-corrected estimates use: propensity_score_matching(),
    difference_in_differences(), or double_machine_learning().

    Returns
    -------
    pd.DataFrame sorted by |Cohen's d| (descending).
    """
    section_header("CAUSAL ANALYSIS (effect sizes)")

    if confounders:
        print(f"  Declared confounders (not adjusted): {confounders}")
        print("  Warning: observational comparison only. Use PSM/DiD/DML for causal inference.")

    records = []

    for col in treatment_cols:
        if col not in df.columns:
            logger.warning("Treatment column '%s' not found — skipped.", col)
            continue

        unique_vals = df[col].dropna().unique()
        if len(unique_vals) != 2:
            logger.debug("'%s' is not binary (%d values) — skipped.", col, len(unique_vals))
            continue

        val0, val1 = sorted(unique_vals)
        group0 = df[df[col] == val0][target_col].dropna()
        group1 = df[df[col] == val1][target_col].dropna()

        if len(group0) < 5 or len(group1) < 5:
            continue

        cohens_d = compute_cohens_d(group0.values, group1.values, variant="pooled")

        try:
            stat, p_val = ttest_ind(group0, group1, equal_var=False)
            test_used = "t-test (Welch)"
        except Exception:
            stat, p_val = mannwhitneyu(group0, group1, alternative="two-sided")
            test_used = "Mann-Whitney U"

        effect_label = (
            "Large"  if abs(cohens_d) > 0.8 else
            "Medium" if abs(cohens_d) > 0.5 else
            "Small"
        )

        records.append({
            "treatment_var": col,
            "group_0_mean":  round(float(group0.mean()), 4),
            "group_1_mean":  round(float(group1.mean()), 4),
            "mean_diff":     round(float(group1.mean() - group0.mean()), 4),
            "cohens_d":      round(cohens_d, 4),
            "p_raw":         round(float(p_val), 6),
            "significant":   None,  # filled after Bonferroni
            "test":          test_used,
            "effect_size":   effect_label,
        })

    if not records:
        print("  No valid binary treatment variable found.")
        return pd.DataFrame()

    causal_df = (
        pd.DataFrame(records)
        .assign(_abs_d=lambda x: x["cohens_d"].abs())
        .sort_values("_abs_d", ascending=False)
        .drop(columns=["_abs_d"])
        .reset_index(drop=True)
    )
    # Bonferroni correction across all treatment variables tested
    p_raws = causal_df["p_raw"].values
    if bonferroni and len(p_raws) > 1:
        p_adj = np.minimum(p_raws * len(p_raws), 1.0)
    else:
        p_adj = p_raws
    causal_df["p_value"]   = p_adj.round(6)
    causal_df["significant"] = causal_df["p_value"] < alpha
    if bonferroni and len(p_raws) > 1:
        print(f"  Bonferroni correction applied across {len(p_raws)} treatment variables.")

    if plot and not causal_df.empty:
        try:
            plot_causal_effect(causal_df, target_col)
        except Exception as exc:
            logger.warning("Causal visualization failed: %s", exc)

    print(f"\n  {len(causal_df)} treatment variable(s) analyzed:")
    print(
        causal_df[["treatment_var", "cohens_d", "p_raw", "p_value", "significant", "effect_size"]]
        .to_string(index=False)
    )
    return causal_df


# ──────────────────────────────────────────────────────────────────────────────
# 10. PROPENSITY SCORE MATCHING
# ──────────────────────────────────────────────────────────────────────────────

def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: List[str],
    caliper: float = 0.05,
    n_neighbors: int = 1,
    random_state: int = 42,
    plot: bool = True,
) -> Dict:
    """
    Propensity Score Matching (PSM) — balance confounders before ATT estimation.

    Steps
    -----
    1. Logistic regression of treatment ~ confounders → propensity scores
    2. Nearest-neighbor matching (with caliper) to pair treated/controls
    3. Balance check: SMD (Standardized Mean Difference) before/after matching
    4. ATT (Average Treatment Effect on the Treated) estimation

    Parameters
    ----------
    caliper          : max absolute PS difference allowed for a match (0 = no caliper)
    n_neighbors      : controls matched per treated unit (default 1:1)

    Returns
    -------
    dict with keys:
        matched_df    : pd.DataFrame of matched pairs
        att           : float — Average Treatment Effect on Treated
        att_pvalue    : float — Welch t-test p-value on matched outcome difference
        balance_before: pd.DataFrame — SMD per confounder before matching
        balance_after : pd.DataFrame — SMD per confounder after matching
        propensity_scores: pd.Series
    """
    section_header("PROPENSITY SCORE MATCHING")

    sub = df[[treatment_col, outcome_col] + confounder_cols].dropna().copy()
    T = sub[treatment_col].astype(int)
    X = sub[confounder_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, random_state=random_state)
        lr.fit(X_scaled, T)
        ps = pd.Series(lr.predict_proba(X_scaled)[:, 1], index=sub.index, name="propensity_score")

    sub["ps"] = ps
    treated   = sub[T == 1].copy()
    controls  = sub[T == 0].copy()

    print(f"\n  Treated: {len(treated)} | Controls: {len(controls)}")
    print(f"  PS range treated:  [{treated['ps'].min():.3f}, {treated['ps'].max():.3f}]")
    print(f"  PS range controls: [{controls['ps'].min():.3f}, {controls['ps'].max():.3f}]")

    # Nearest-neighbor matching with caliper
    
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(controls[["ps"]].values)
    distances, indices = nn.kneighbors(treated[["ps"]].values)

    matched_pairs = []
    used_controls = set()
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        treated_row = treated.iloc[i]
        for dist, idx in zip(dists, idxs):
            ctrl_idx = controls.index[idx]
            if caliper > 0 and dist > caliper:
                continue
            if ctrl_idx in used_controls:
                continue
            ctrl_row = controls.loc[ctrl_idx]
            matched_pairs.append({
                "treated_idx": treated_row.name,
                "control_idx": ctrl_idx,
                "treated_outcome": float(treated_row[outcome_col]),
                "control_outcome": float(ctrl_row[outcome_col]),
                "ps_diff": round(float(dist), 5),
            })
            used_controls.add(ctrl_idx)
            break

    n_matched = len(matched_pairs)
    print(f"\n  Matched pairs: {n_matched} / {len(treated)} treated units")

    if n_matched < 5:
        print("  Insufficient matches. Relax caliper or add more controls.")
        return {}

    matched_df = pd.DataFrame(matched_pairs)
    treated_out = matched_df["treated_outcome"].values
    control_out = matched_df["control_outcome"].values
    att = float(treated_out.mean() - control_out.mean())
    
    _, att_p = _ttest_rel(treated_out, control_out)

    # Balance assessment (SMD)
    def _smd(series_t, series_c):
        """Standardized Mean Difference — Austin (2011) pooled-variance formula."""
        nt, nc = len(series_t), len(series_c)
        if nt < 2 or nc < 2:
            return 0.0
        st = float(series_t.std(ddof=1))
        sc = float(series_c.std(ddof=1))
        pooled_var = ((nt - 1) * st ** 2 + (nc - 1) * sc ** 2) / (nt + nc - 2)
        pooled_sd = float(np.sqrt(max(pooled_var, 1e-12)))
        return float((series_t.mean() - series_c.mean()) / pooled_sd)

    balance_rows = []
    matched_t_idx = matched_df["treated_idx"].values
    matched_c_idx = matched_df["control_idx"].values
    for col in confounder_cols:
        smd_before = _smd(treated[col], controls[col])
        smd_after  = _smd(
            sub.loc[matched_t_idx, col],
            sub.loc[matched_c_idx, col],
        )
        balance_rows.append({
            "confounder": col,
            "smd_before": round(smd_before, 4),
            "smd_after":  round(smd_after, 4),
            "improved":   bool(abs(smd_after) < abs(smd_before)),
        })
    balance_df = pd.DataFrame(balance_rows)

    print(f"\n  ATT = {att:.4f}  (p = {att_p:.4f}, paired t-test)")
    print(f"\n  Balance check (SMD < 0.1 = well-balanced):\n")
    print(balance_df.to_string(index=False))

    if plot:
        try:
            propensity_score_matching_plot(controls,treated,outcome_col,balance_df,att, att_p)
        except Exception as exc:
            logger.debug("PSM plot failed: %s", exc)
        

    return {
        "matched_df":        matched_df,
        "att":               att,
        "att_pvalue":        float(att_p),
        "balance_before":    balance_df[["confounder", "smd_before"]],
        "balance_after":     balance_df[["confounder", "smd_after"]],
        "propensity_scores": ps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 11. DIFFERENCE-IN-DIFFERENCES
# ──────────────────────────────────────────────────────────────────────────────

def difference_in_differences(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    pre_value,
    post_value,
    covariates: Optional[List[str]] = None,
    plot: bool = True,
) -> Dict:
    """
    Difference-in-Differences (DiD) estimator.

    Compares the change in outcome between treated and control groups
    from pre-period to post-period.

    DiD = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)

    Uses OLS regression: outcome ~ treatment + time + treatment*time [+ covariates]

    Parameters
    ----------
    pre_value  : value of time_col identifying the pre-period
    post_value : value of time_col identifying the post-period

    Returns
    -------
    dict with keys: did_estimate, se, p_value, ci_95, parallel_trend_check
    """
    section_header("DIFFERENCE-IN-DIFFERENCES")

    sub = df[df[time_col].isin([pre_value, post_value])].copy()
    sub = sub[[outcome_col, treatment_col, time_col]
              + (covariates or [])].dropna()
    sub["post"] = (sub[time_col] == post_value).astype(int)
    sub["treated"] = sub[treatment_col].astype(int)
    sub["interaction"] = sub["post"] * sub["treated"]

    feature_cols = ["treated", "post", "interaction"] + (covariates or [])
    X = sub[feature_cols].values
    y = sub[outcome_col].values

    try:
        X_aug = np.column_stack([np.ones(len(X)), X])
        coeffs, _, _, _ = lstsq(X_aug, y, rcond=None)
        did_estimate = float(coeffs[feature_cols.index("interaction") + 1])

        y_pred = X_aug @ coeffs
        resid  = y - y_pred
        n, k   = X_aug.shape
        
        h_diag = np.einsum("ij,jk,ki->i", X_aug, np.linalg.pinv(X_aug.T @ X_aug), X_aug.T)
        hc3_weights = (resid / np.maximum(1 - h_diag, 1e-10)) ** 2
        meat   = X_aug.T @ np.diag(hc3_weights) @ X_aug
        bread  = np.linalg.pinv(X_aug.T @ X_aug)
        vcov   = bread @ meat @ bread
        int_idx = feature_cols.index("interaction") + 1
        se     = float(np.sqrt(max(vcov[int_idx, int_idx], 0)))
        t_stat = did_estimate / (se + 1e-12)
        p_val  = float(2 * t_dist.sf(abs(t_stat), df=n - k))
        ci_lo  = did_estimate - 1.96 * se
        ci_hi  = did_estimate + 1.96 * se
    except (LinAlgError, Exception) as exc:
        logger.warning("DiD regression failed: %s", exc)
        return {}

    pre_data = sub[sub["post"] == 0]
    t_pre = pre_data[pre_data["treated"] == 1][outcome_col].mean()
    c_pre = pre_data[pre_data["treated"] == 0][outcome_col].mean()
    t_post = sub[sub["post"] == 1][sub["treated"] == 1][outcome_col].mean()
    c_post = sub[sub["post"] == 1][sub["treated"] == 0][outcome_col].mean()

    parallel_trend_note = (
        "Only one pre-period time point available — parallel trends cannot be "
        "formally tested (requires ≥ 2 pre-period observations per group). "
        "Assumption must be justified on domain knowledge."
    )
    # If the time_col has numeric/datetime values with multiple pre-period points,
    # we can perform the slope test.
    pre_unique_times = sub.loc[sub["post"] == 0, time_col].nunique()
    if pre_unique_times >= 2:
        # Interaction of time × treated in pre-period: coefficient tests parallel slope
        pre_sub    = sub[sub["post"] == 0].copy()
        
        pre_sub = pre_sub.copy()
        try:
            pre_sub = pre_sub.sort_values(time_col)
        except TypeError:
            pass  # non-sortable time_col — factorize order may be arbitrary
        pre_sub["time_numeric"] = pd.factorize(pre_sub[time_col].astype(str).str.strip())[0].astype(float)
        # Verify the encoding is monotone with the sorted order
        _time_order = pre_sub[[time_col, "time_numeric"]].drop_duplicates().sort_values("time_numeric")
        logger.debug("Parallel trends time encoding: %s", _time_order.values.tolist())
        X_pt = np.column_stack([
            np.ones(len(pre_sub)),
            pre_sub["time_numeric"].values,
            pre_sub["treated"].values,
            pre_sub["time_numeric"].values * pre_sub["treated"].values,
        ])
        y_pt = pre_sub[outcome_col].values
        try:
            c_pt, _, _, _ = lstsq(X_pt, y_pt, rcond=None)
            slope_diff = float(c_pt[3])
            resid_pt = y_pt - X_pt @ c_pt
            n_pt, k_pt = X_pt.shape
            h_pt = X_pt @ np.linalg.pinv(X_pt.T @ X_pt) @ X_pt.T
            h_d  = np.diag(h_pt)
            w_pt = (resid_pt / np.maximum(1 - h_d, 1e-10)) ** 2
            vcov_pt = np.linalg.pinv(X_pt.T @ X_pt) @ (X_pt.T @ np.diag(w_pt) @ X_pt) @ np.linalg.pinv(X_pt.T @ X_pt)
            se_pt   = float(np.sqrt(max(vcov_pt[3, 3], 0)))
            t_pt    = slope_diff / (se_pt + 1e-12)
            p_pt    = float(2 * t_dist.sf(abs(t_pt), df=n_pt - k_pt))
            parallel_trend_note = (
                f"Parallel trends test (slope diff = {slope_diff:.4f}, "
                f"p = {p_pt:.4f}) — "
                + ("ASSUMPTION VIOLATED (p < 0.05)" if p_pt < 0.05
                   else "assumption NOT rejected (p ≥ 0.05)")
            )
            print(f"\n  Parallel trends test: slope_diff={slope_diff:.4f}, p={p_pt:.4f}")
            if p_pt < 0.05:
                print("  Parallel trends VIOLATED — DiD estimate may be biased.")
        except Exception as exc:
            logger.debug("Parallel trends slope test failed: %s", exc)
    else:
        print(f"\n  Parallel trends: {parallel_trend_note}")

    print(f"\n  DiD Estimate: {did_estimate:.4f}  SE (HC3): {se:.4f}")
    print(f"  p-value:  {p_val:.4f}  |  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\n  Pre-period  — Treated: {t_pre:.4f} | Control: {c_pre:.4f}")
    print(f"  Post-period — Treated: {t_post:.4f} | Control: {c_post:.4f}")
    print(f"  DiD = ({t_post:.4f} - {t_pre:.4f}) - ({c_post:.4f} - {c_pre:.4f}) = {did_estimate:.4f}")

    if plot:
        try:
            difference_in_differences_plot(pre_value, post_value, t_pre, t_post, c_pre, c_post, did_estimate, outcome_col)
        except Exception as exc:
            logger.debug("DiD plot failed: %s", exc)
       

    return {
        "did_estimate":          did_estimate,
        "se":                    se,
        "se_type":               "HC3",
        "p_value":               p_val,
        "ci_95":                 (ci_lo, ci_hi),
        "t_pre":                 t_pre,
        "t_post":                t_post,
        "c_pre":                 c_pre,
        "c_post":                c_post,
        "parallel_trend_check":  parallel_trend_note,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 12. REGRESSION DISCONTINUITY
# ──────────────────────────────────────────────────────────────────────────────

def regression_discontinuity(
    df: pd.DataFrame,
    outcome_col: str,
    running_col: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
    plot: bool = True,
) -> Dict:
    """
    Regression Discontinuity Design (RDD) — local linear estimation.

    Compares units just below and just above a threshold (cutoff) on a
    running variable to estimate the causal effect of crossing that threshold.

    Parameters
    ----------
    running_col : the forcing/assignment variable (e.g., exam score, age)
    cutoff      : threshold value that determines treatment assignment
    bandwidth   : window around cutoff to use (IK-style; auto if None)

    Returns
    -------
    dict with keys: rdd_estimate, se, p_value, ci_95, n_left, n_right
    """
    section_header("REGRESSION DISCONTINUITY")

    sub = df[[outcome_col, running_col]].dropna().copy()
    sub["treated"] = (sub[running_col] >= cutoff).astype(int)
    sub["x_centered"] = sub[running_col] - cutoff

    # Auto bandwidth: 0.5 × IQR of running variable
    if bandwidth is None:
        iqr = float(sub["x_centered"].quantile(0.75) - sub["x_centered"].quantile(0.25))
        bandwidth = max(iqr * 0.5, float(sub["x_centered"].abs().quantile(0.1)))
        print(f"  Auto bandwidth: ±{bandwidth:.4f}")

    local = sub[sub["x_centered"].abs() <= bandwidth].copy()
    n_left  = int((local["treated"] == 0).sum())
    n_right = int((local["treated"] == 1).sum())

    print(f"  Cutoff: {cutoff}  |  Bandwidth: ±{bandwidth:.4f}")
    print(f"  Obs left: {n_left}  |  Obs right: {n_right}")

    if n_left < 5 or n_right < 5:
        print("  Insufficient observations within bandwidth.")
        return {}

    # Local linear regression: Y ~ x_centered * treated
    X = np.column_stack([
        np.ones(len(local)),
        local["x_centered"].values,
        local["treated"].values,
        local["x_centered"].values * local["treated"].values,
    ])
    y = local[outcome_col].values

    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    rdd_estimate = float(coeffs[2])  # coefficient on "treated"

    y_pred = X @ coeffs
    resid  = y - y_pred
    n, k   = X.shape
   
    try:
        h_diag_rdd = np.einsum("ij,jk,ki->i", X, np.linalg.pinv(X.T @ X), X.T)
        hc3_w_rdd  = (resid / np.maximum(1 - h_diag_rdd, 1e-10)) ** 2
        meat_rdd   = X.T @ np.diag(hc3_w_rdd) @ X
        bread_rdd  = np.linalg.pinv(X.T @ X)
        vcov_rdd   = bread_rdd @ meat_rdd @ bread_rdd
        se = float(np.sqrt(max(vcov_rdd[2, 2], 0)))
    except Exception:
        # Fallback to OLS SE with warning
        sigma2 = float(np.sum(resid ** 2) / max(n - k, 1))
        try:
            se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[2, 2]))
        except Exception:
            se = 0.0
        logger.warning("RDD: HC3 SE failed — using OLS SE (may be anti-conservative).")

    t_stat = rdd_estimate / (se + 1e-12)
    p_val  = float(2 * t_dist.sf(abs(t_stat), df=n - k))
    ci_lo  = rdd_estimate - 1.96 * se
    ci_hi  = rdd_estimate + 1.96 * se

    print(f"\n  RDD Estimate: {rdd_estimate:.4f}  SE (HC3): {se:.4f}")
    print(f"  p-value: {p_val:.4f}  |  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    if plot:
        try:
            regression_discontinuity_plot(local,coeffs,cutoff,rdd_estimate,p_val, running_col, outcome_col,y)
        except Exception as exc:
            logger.debug("RDD plot failed: %s", exc)
        

    return {
        "rdd_estimate": rdd_estimate,
        "se":           se,
        "se_type":      "HC3",
        "p_value":      p_val,
        "ci_95":        (ci_lo, ci_hi),
        "n_left":       n_left,
        "n_right":      n_right,
        "bandwidth":    bandwidth,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 13. CAUSAL FOREST  (Honest splitting — GRF-style ATE)
# ──────────────────────────────────────────────────────────────────────────────

def causal_forest_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    feature_cols: Optional[List[str]] = None,
    n_estimators: int = 200,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    plot: bool = True,
) -> Dict:
    """
    Causal Forest — Heterogeneous Treatment Effect estimation.

    Implements a simplified honest Causal Forest using two-step residualization
    (Robinson 1988 / Wager-Athey 2018 approximation):
      1. Fit E[Y|X] and E[T|X] using separate Random Forests
      2. Residualize: Ỹ = Y - E[Y|X],  T̃ = T - E[T|X]
      3. Fit RF on Ỹ ~ T̃ * X to estimate CATE (Conditional ATE)

    Returns ATE = mean(CATE) with bootstrap confidence interval.

    Parameters
    ----------
    feature_cols : confounders/moderators (auto-detected if None)

    Returns
    -------
    dict with keys: ate, ate_ci_95, cate_series, top_heterogeneity_features
    """
    section_header("CAUSAL FOREST (Honest ATE Estimation)")
    sub = df[[outcome_col, treatment_col]
              + (feature_cols or [])].dropna().copy()

    T = sub[treatment_col].astype(float).values
    Y = sub[outcome_col].astype(float).values

    if feature_cols is None:
        feature_cols = [
            c for c in sub.select_dtypes(include=[np.number]).columns
            if c not in [outcome_col, treatment_col]
        ]
    if not feature_cols:
        print("  No numeric feature columns found for causal forest.")
        return {}

    X = sub[feature_cols].fillna(sub[feature_cols].median()).values
    n = len(Y)
    print(f"\n  N={n}  |  Treated={int(T.sum())}  |  Control={int((1-T).sum())}")
    print(f"  Features: {feature_cols}")

    is_binary_t = len(np.unique(T)) <= 2

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # E[Y|X]
            rf_y = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                          random_state=random_state, n_jobs=-1)
            Y_hat = cross_val_predict(rf_y, X, Y, cv=5)

            # E[T|X]
            if is_binary_t:
                rf_t = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                               random_state=random_state, n_jobs=-1)
                T_hat = cross_val_predict(rf_t, X, T.astype(int), cv=5, method="predict_proba")[:, 1]
            else:
                rf_t = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                              random_state=random_state, n_jobs=-1)
                T_hat = cross_val_predict(rf_t, X, T, cv=5)

            # Residualize
            Y_res = Y - Y_hat
            T_res = T - T_hat

            # CATE via weighted RF
            weights = T_res ** 2
            rf_cate = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                             random_state=random_state, n_jobs=-1)
           
            T_res_clipped = np.where(
                np.abs(T_res) < 0.01,
                np.sign(T_res + 1e-12) * 0.01,
                T_res,
            )
            pseudo_outcome = Y_res / T_res_clipped
            rf_cate.fit(X, pseudo_outcome, sample_weight=weights)
            cate = rf_cate.predict(X)

        ate = float(cate.mean())

        ate_var = float(np.var(cate, ddof=1) / n)
        z975 = float(_norm_cf.ppf(0.975))
        ci_lo = ate - z975 * float(np.sqrt(ate_var))
        ci_hi = ate + z975 * float(np.sqrt(ate_var))
        print(f"  95% CI (influence-fn): [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(
            "  NOTE: CI based on Var(CATE)/n approximation (Wager & Athey 2018)."
            " For exact CI use the econml or grf packages."
        )

        # Top heterogeneity features
        feat_imp = pd.Series(rf_cate.feature_importances_, index=feature_cols).sort_values(ascending=False)

        cate_series = pd.Series(cate, index=sub.index, name="cate")

        print(f"\n  ATE  = {ate:.4f}")
        print(f"\n  Top heterogeneity drivers:")
        print(feat_imp.head(5).to_string())

    except Exception as exc:
        logger.warning("Causal Forest failed: %s", exc)
        return {}

    if plot:
        try:
            causal_forest_plot(cate, feat_imp, ate, ci_lo, ci_hi)
        except Exception as exc:
            logger.debug("Causal Forest plot failed: %s", exc)
       

    return {
        "ate":                       ate,
        "ate_ci_95":                 (ci_lo, ci_hi),
        "cate_series":               cate_series,
        "top_heterogeneity_features": feat_imp.head(5).to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 14. DOUBLE MACHINE LEARNING (DML — Robinson 1988 / Chernozhukov 2018)
# ──────────────────────────────────────────────────────────────────────────────

def double_machine_learning(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    confounder_cols: List[str],
    n_folds: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
    plot: bool = True,
) -> Dict:
    """
    Double Machine Learning (DML) — Partialling-Out estimator.

    Debiases ATE by removing confounding bias via cross-fitted nuisance functions:
      1. Fit E[Y|X] and E[T|X] using cross-fitting (n_folds)
      2. Residualize: Ỹ = Y − Ê[Y|X],  T̃ = T − Ê[T|X]
      3. ATE = OLS of Ỹ ~ T̃  (Robinson 1988)

    Cross-fitting prevents regularization bias from inflating the treatment estimate.

    Returns
    -------
    dict with keys: ate, se, p_value, ci_95, r2_outcome_nuisance, r2_treatment_nuisance
    """
    section_header("DOUBLE MACHINE LEARNING (DML)")
    sub = df[[outcome_col, treatment_col] + confounder_cols].dropna().copy()
    Y = sub[outcome_col].astype(float).values
    T = sub[treatment_col].astype(float).values
    X = sub[confounder_cols].fillna(sub[confounder_cols].median()).values
    n = len(Y)
    is_binary_t = len(np.unique(T)) <= 2

    print(f"\n  N={n}  |  Confounders: {confounder_cols}")
    print(f"  Cross-fitting folds: {n_folds}")

    Y_res = np.zeros(n)
    T_res = np.zeros(n)

    # P3-1 OPTIMIZATION NOTE: the K cross-fitting folds below are independent
    # and could be parallelized via joblib.Parallel(n_jobs=-1). At n_folds=5
    # with 100-tree forests this is not a bottleneck for n < 50k, but for
    # larger datasets consider:
    #   from joblib import Parallel, delayed
    #   results = Parallel(n_jobs=-1)(delayed(_fit_fold)(X, Y, T, tr, te) for tr, te in kf.split(X))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            rf_y = RandomForestRegressor(n_estimators=n_estimators, max_depth=8,
                                          random_state=random_state, n_jobs=-1)
            rf_y.fit(X[train_idx], Y[train_idx])
            Y_res[test_idx] = Y[test_idx] - rf_y.predict(X[test_idx])

            if is_binary_t:
                rf_t = RandomForestClassifier(n_estimators=n_estimators, max_depth=8,
                                               random_state=random_state, n_jobs=-1)
                rf_t.fit(X[train_idx], T[train_idx].astype(int))
                T_res[test_idx] = T[test_idx] - rf_t.predict_proba(X[test_idx])[:, 1]
            else:
                rf_t = RandomForestRegressor(n_estimators=n_estimators, max_depth=8,
                                              random_state=random_state, n_jobs=-1)
                rf_t.fit(X[train_idx], T[train_idx])
                T_res[test_idx] = T[test_idx] - rf_t.predict(X[test_idx])

    # Final OLS: Ỹ ~ T̃
    T_res_sq_sum = float(np.sum(T_res ** 2))
    if T_res_sq_sum < 1e-12:
        print("  T residuals near zero — treatment may be fully explained by confounders.")
        return {}

    ate = float(np.sum(T_res * Y_res) / T_res_sq_sum)

    # Variance via influence-function sandwich SE (Chernozhukov et al. 2018).
    # The correct formula for the partialling-out estimator is:
    #   Var(ATE) = (1/n²) × Σψᵢ² / (Σ T̃ᵢ²/n)²
    # where ψᵢ = T̃ᵢ × (Ỹᵢ - ATE × T̃ᵢ) is the influence function.
    # Previous code used V/n where V = Σψ²/(ΣT̃²)² — mixing normalizations.
    psi    = T_res * (Y_res - ate * T_res)       # influence function per obs
    sigma2_psi = float(np.mean(psi ** 2))         # E[ψᵢ²]
    T_res_var  = float(np.mean(T_res ** 2))       # E[T̃ᵢ²]
    se = float(np.sqrt(sigma2_psi / (n * T_res_var ** 2)))

    t_stat = ate / (se + 1e-12)
    p_val  = float(2 * t_dist.sf(abs(t_stat), df=n - len(confounder_cols) - 1))
    ci_lo  = ate - 1.96 * se
    ci_hi  = ate + 1.96 * se

    # Nuisance quality
    r2_y = float(1 - np.var(Y_res) / (np.var(Y) + 1e-12))
    r2_t = float(1 - np.var(T_res) / (np.var(T) + 1e-12))

    print(f"\n  ATE  (DML) = {ate:.4f}")
    print(f"  SE         = {se:.4f}")
    print(f"  p-value    = {p_val:.4f}")
    print(f"  95% CI     = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\n  Nuisance R² — outcome: {r2_y:.3f}  |  treatment: {r2_t:.3f}")

    if plot:
        try:
            double_machine_learning_plot(T_res, Y_res, ate, ci_lo, ci_hi, p_val)
        except Exception as exc:
            logger.debug("DML plot failed: %s", exc)
       
    return {
        "ate":                     ate,
        "se":                      se,
        "p_value":                 p_val,
        "ci_95":                   (ci_lo, ci_hi),
        "r2_outcome_nuisance":     r2_y,
        "r2_treatment_nuisance":   r2_t,
        "Y_residuals":             Y_res,
        "T_residuals":             T_res,
    }
