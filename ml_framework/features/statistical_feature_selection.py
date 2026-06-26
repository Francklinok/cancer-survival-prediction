"""
statistical_feature_selection.py — Feature selection via statistical significance tests.

Methods:
  - ANOVA / Kruskal-Wallis (numeric × categorical target)
  - Pearson / Spearman (numeric × continuous target)
  - Chi-square (categorical × categorical target)
  - Mutual Information (all configurations)

Robustness:
  - ANOVA assumption validation (Shapiro-Wilk normality + Levene homoscedasticity)
  - Automatic fallback to Kruskal-Wallis if assumptions are violated
  - Multiple testing corrections (FDR Benjamini-Hochberg, Bonferroni, Holm)
  - Scientific MI threshold: percentile / top_k / relative

Result: detailed DataFrame + list of significant features
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    pearsonr,
    spearmanr,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.multitest import multipletests

from ml_framework.utils.display_utils import section_header
from ml_framework.utils.statistical_utils import (
    all_groups_normal,
    groups_have_equal_variance,
)

logger = logging.getLogger("ml_framework.stat_feature_selection")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def statistical_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cat_columns: Optional[List[str]] = None,
    num_columns: Optional[List[str]] = None,
    test_type: str = "auto",
    significance_level: float = 0.05,
    correction_method: str = "fdr_bh",
    mi_selection: str = "percentile",
    mi_percentile: float = 50.0,
    mi_top_k: Optional[int] = None,
    normality_alpha: float = 0.05,
    homoscedasticity_alpha: float = 0.05,
    verbose: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Feature selection via statistical significance tests.

    Parameters
    ----------
    X                      : feature matrix
    y                      : target series
    cat_columns            : categorical columns (auto-detected if None)
    num_columns            : numeric columns (auto-detected if None)
    test_type              : 'auto' | 'anova' | 'kruskal' | 'chi2' |
                             'pearson' | 'spearman' | 'mutual_info'
    significance_level     : alpha threshold after correction
    correction_method      : 'bonferroni' | 'fdr_bh' | 'holm' | None
    mi_selection           : 'percentile' | 'top_k' | 'relative'
    mi_percentile          : percentile for MI threshold (if mi_selection='percentile')
    mi_top_k               : number of features to select (if mi_selection='top_k')
    normality_alpha        : Shapiro-Wilk significance threshold
    homoscedasticity_alpha : Levene significance threshold

    Returns
    -------
    significant_features : List[str]
    results_df           : pd.DataFrame — full detail
    """
    if cat_columns is None and num_columns is None:
        num_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_columns = [c for c in X.columns if c not in num_columns]
    cat_columns = cat_columns or []
    num_columns = num_columns or []

    target_is_numeric = pd.api.types.is_numeric_dtype(y)
    target_is_categorical = not target_is_numeric or y.nunique() <= 20

    records: list = []

    for col in num_columns:
        chosen_test = _select_test(test_type, "numeric", target_is_categorical)
        row = {
            "feature": col, "type": "numeric", "test_used": chosen_test,
            "stat": np.nan, "p_raw": np.nan, "note": "",
        }

        try:
            if chosen_test in ("anova", "kruskal") and target_is_categorical:
                groups = [X[col][y == v].dropna().values for v in y.unique()]
                groups = [g for g in groups if len(g) >= 3]

                if len(groups) < 2:
                    row["note"] = "< 2 valid groups"
                    records.append(row)
                    continue

                use_kruskal = chosen_test == "kruskal"

                if chosen_test == "anova":
                    normality_ok = all_groups_normal(groups, alpha=normality_alpha)
                    homoscedasticity_ok = True
                    if normality_ok:
                        _, _, homoscedasticity_ok = groups_have_equal_variance(
                            groups, alpha=homoscedasticity_alpha
                        )

                    if not normality_ok or not homoscedasticity_ok:
                        use_kruskal = True
                        notes = []
                        if not normality_ok:
                            notes.append("non-normal")
                        if not homoscedasticity_ok:
                            notes.append("heteroscedastic")
                        row["note"] = f"ANOVA→Kruskal [{', '.join(notes)}]"
                        row["test_used"] = "kruskal"

                if use_kruskal:
                    stat, p_val = kruskal(*groups)
                else:
                    stat, p_val = f_oneway(*groups)
                    row["test_used"] = "anova"

                row["stat"] = stat
                row["p_raw"] = p_val

            elif chosen_test in ("pearson", "spearman") and not target_is_categorical:
                valid = X[[col]].join(y.rename("__y__")).dropna()
                x_v, y_v = valid[col].values, valid["__y__"].values

                if chosen_test == "pearson":
                    stat, p_val = pearsonr(x_v, y_v)
                else:
                    stat, p_val = spearmanr(x_v, y_v)

                row["stat"] = stat
                row["p_raw"] = p_val

            elif chosen_test == "mutual_info":
                row["note"] = "MI — grouped processing"

            else:
                row["note"] = f"Test '{chosen_test}' not applicable"

        except Exception as e:
            row["note"] = f"Error: {e}"

        records.append(row)

    for col in cat_columns:
        chosen_test = _select_test(test_type, "categorical", target_is_categorical)
        row = {
            "feature": col, "type": "categorical", "test_used": chosen_test,
            "stat": np.nan, "p_raw": np.nan, "note": "",
        }

        try:
            if chosen_test == "chi2" and target_is_categorical:
                contingency = pd.crosstab(X[col], y)
                chi2_stat, p_val, dof, _ = chi2_contingency(contingency)
                row["stat"] = chi2_stat
                row["p_raw"] = p_val
                row["note"] = f"dof={dof}"

            elif chosen_test == "mutual_info":
                row["note"] = "MI — grouped processing"

            else:
                row["note"] = f"Test '{chosen_test}' not applicable"

        except Exception as e:
            row["note"] = f"Error: {e}"

        records.append(row)

    mi_cols_num = [r["feature"] for r in records if r["test_used"] == "mutual_info" and r["type"] == "numeric"]
    mi_cols_cat = [r["feature"] for r in records if r["test_used"] == "mutual_info" and r["type"] == "categorical"]

    if test_type == "mutual_info":
        mi_cols_num = num_columns
        mi_cols_cat = cat_columns

    all_mi_cols = mi_cols_num + mi_cols_cat 

    if all_mi_cols:
        records = _compute_mutual_info(
            records, X, y, all_mi_cols, mi_cols_cat,
            mi_selection, mi_percentile, mi_top_k,
            target_is_numeric, target_is_categorical,
        )

    results_df = pd.DataFrame(records)
    results_df["p_corrected"] = np.nan
    results_df["significant"] = False

    mask_pval = results_df["p_raw"].notna() & (results_df["test_used"] != "mutual_info")

    if mask_pval.sum() > 0 and correction_method is not None:
        p_raw_array = results_df.loc[mask_pval, "p_raw"].values
        reject, p_corrected, _, _ = multipletests(
            p_raw_array, alpha=significance_level, method=correction_method
        )
        results_df.loc[mask_pval, "p_corrected"] = p_corrected
        results_df.loc[mask_pval, "significant"] = reject

    elif mask_pval.sum() > 0:
        results_df.loc[mask_pval, "p_corrected"] = results_df.loc[mask_pval, "p_raw"]
        results_df.loc[mask_pval, "significant"] = (
            results_df.loc[mask_pval, "p_raw"] <= significance_level
        )

    if "mi_selected" in results_df.columns:
        mi_mask = results_df["test_used"] == "mutual_info"
        results_df.loc[mi_mask, "significant"] = results_df.loc[mi_mask, "mi_selected"]

    significant_features = results_df.loc[results_df["significant"], "feature"].tolist()

    if verbose:
        _print_selection_summary(results_df, significant_features, correction_method)

    return significant_features, results_df


# =============================================================================
# HELPERS
# =============================================================================


def _select_test(test_type: str, col_type: str, target_is_categorical: bool) -> str:
    if test_type != "auto":
        return test_type

    if col_type == "numeric":
        return "anova" if target_is_categorical else "spearman"
    else:
        return "chi2" if target_is_categorical else "mutual_info"


def _compute_mutual_info(
    records, X, y, all_mi_cols, mi_cols_cat,
    mi_selection, mi_percentile, mi_top_k,
    target_is_numeric, target_is_categorical,
):
    X_mi = X[all_mi_cols]
    discrete_mask = [c in mi_cols_cat for c in all_mi_cols]

    mi_fn = (
        mutual_info_regression
        if (target_is_numeric and not target_is_categorical)
        else mutual_info_classif
    )
    mi_scores = mi_fn(X_mi, y, discrete_features=discrete_mask, random_state=42)
    mi_series = pd.Series(mi_scores, index=all_mi_cols)

    if mi_selection == "percentile":
        threshold = float(np.percentile(mi_scores, mi_percentile))
    elif mi_selection == "top_k":
        k = mi_top_k or max(1, len(all_mi_cols) // 2)
        threshold = float(sorted(mi_scores, reverse=True)[k - 1])
    elif mi_selection == "relative":
        threshold = 0.10 * float(mi_scores.max())
    else:
        raise ValueError("mi_selection must be 'percentile', 'top_k', or 'relative'.")

    for r in records:
        feat = r["feature"]
        if feat in mi_series.index:
            r["stat"] = float(mi_series[feat])
            r["p_raw"] = np.nan
            r["note"] += f" MI={mi_series[feat]:.4f} threshold={threshold:.4f}"
            r["mi_selected"] = bool(mi_series[feat] >= threshold)

    return records


def _print_selection_summary(
    results_df: pd.DataFrame,
    significant_features: List[str],
    correction_method: Optional[str],
) -> None:
    section_header("STATISTICAL FEATURE SELECTION", width=62)
    n_total = len(results_df)
    n_sig = len(significant_features)
    print(f"  Features analyzed    : {n_total}")
    print(f"  Significant features : {n_sig} ({n_sig/n_total*100:.1f}%)")
    print(f"  Correction applied   : {correction_method or 'none'}")
    print()

    if significant_features:
        print("  Selected features:")
        for f in significant_features:
            row = results_df[results_df["feature"] == f].iloc[0]
            p_str = (
                f"p_corr={row['p_corrected']:.4f}"
                if not np.isnan(row.get("p_corrected", np.nan))
                else f"MI={row['stat']:.4f}"
            )
            print(f"    ✓ {f:<35} [{row['test_used']}]  {p_str}")
    else:
        print(" No significant features after correction.")

    print("═" * 62 + "\n")
