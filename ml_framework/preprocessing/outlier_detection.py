"""
outlier_detection.py — Multi-method outlier detection.

Detection methods:
  - IQR (Tukey)
  - Z-score
  - Modified Z-score (median/MAD — robust)
  - IsolationForest (unsupervised ML)
  - LocalOutlierFactor
  - EllipticEnvelope

Available treatments:
  - Winsorization (clip to bounds)
  - Removal (drop rows)
  - Median imputation
  - Flag indicator column

Outputs:
  - Per-column dict with metrics and masks
  - Multi-method pivot table
  - Visualizations
  - Automatic interpretations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("ml_framework.outlier_detection")

_METHODS = (
    "iqr",
    "zscore",
    "modified_zscore",
    "IsolationForest",
    "LocalOutlierFactor",
    "EllipticEnvelope",
)


# ──────────────────────────────────────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────────────────────────────────────


def identify_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
    contamination: float = 0.05,
    verbose: bool = False,
    return_mask: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Identify outliers in each numeric column.

    Parameters
    ----------
    df            : source DataFrame
    columns       : columns to analyze (all non-binary numeric if None)
    method        : 'iqr' | 'zscore' | 'modified_zscore' |
                    'IsolationForest' | 'LocalOutlierFactor' | 'EllipticEnvelope'
    threshold     : IQR multiplier (1.5 = standard) or z-score threshold (3.0)
    contamination : expected outlier proportion (ML methods)
    verbose       : print a summary table
    return_mask   : include boolean mask in the result dict

    Returns
    -------
    dict[col_name → dict{count, percentage, lower_bound, upper_bound,
                         indices, (mask)}]
    """
    if method not in _METHODS:
        raise ValueError(f"Method '{method}' not supported. Choose from: {_METHODS}")

    if columns is None:
        columns     = df.select_dtypes(include=[np.number]).columns.tolist()
        binary_cols = [c for c in columns if df[c].dropna().isin([0, 1]).all()]
        columns     = [c for c in columns if c not in binary_cols]

    outliers: Dict[str, Dict[str, Any]] = {}
    n    = len(df)
    data = df[columns].copy()

    # Pre-compute statistical scores
    z_scores = None
    mod_z    = None

    if method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std().replace(0, np.nan))

    elif method == "modified_zscore":
        median = data.median()
        mad    = (np.abs(data - median)).median().replace(0, np.nan)
        mod_z  = 0.6745 * (data - median) / mad

    # ML methods
    ml_mask = None

    if method in ("IsolationForest", "LocalOutlierFactor", "EllipticEnvelope"):
        X = data.dropna()

        if len(X) < 10:
            logger.warning("Not enough data for ML method (%d rows).", len(X))
            return {}

        try:
            if method == "IsolationForest":
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

            elif method == "LocalOutlierFactor":
                from sklearn.neighbors import LocalOutlierFactor
                model = LocalOutlierFactor(
                    n_neighbors=min(20, len(X) - 1),
                    contamination=contamination,
                    n_jobs=-1,
                )

            elif method == "EllipticEnvelope":
                from sklearn.covariance import EllipticEnvelope
                model = EllipticEnvelope(contamination=contamination, random_state=42)

            preds   = model.fit_predict(X)
            ml_mask = pd.Series(False, index=df.index)
            ml_mask.loc[X.index] = preds == -1

        except Exception as exc:
            logger.error("ML method '%s' failed: %s", method, exc)
            return {}

    # Per-column loop
    for col in columns:
        col_series  = df[col].dropna()
        if col_series.nunique() <= 1:
            continue

        lower_bound: Optional[float] = None
        upper_bound: Optional[float] = None

        if method == "iqr":
            q1          = float(col_series.quantile(0.25))
            q3          = float(col_series.quantile(0.75))
            iqr         = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask        = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == "zscore":
            mask = z_scores[col] > threshold

        elif method == "modified_zscore":
            mask = np.abs(mod_z[col]) > 3.5

        elif ml_mask is not None:
            mask = ml_mask.reindex(df.index, fill_value=False)

        else:
            continue

        mask  = mask.fillna(False)
        count = int(mask.sum())

        entry: Dict[str, Any] = {
            "count":       count,
            "percentage":  round(count / n * 100, 3),
            "lower_bound": round(lower_bound, 6) if lower_bound is not None else None,
            "upper_bound": round(upper_bound, 6) if upper_bound is not None else None,
            "indices":     df.index[mask].tolist(),
        }

        if return_mask:
            entry["mask"] = mask

        outliers[col] = entry

    if verbose:
        _print_outlier_summary(outliers, method)

    return outliers


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-METHOD PIVOT TABLE
# ──────────────────────────────────────────────────────────────────────────────


def outlier_pivot_table(
    df: pd.DataFrame,
    methods: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Compute outliers for all methods and return a pivot table.

    Returns
    -------
    pd.DataFrame — index=column, columns={method}_{metric}
    """
    default_methods = {m: m for m in _METHODS}
    method_map      = methods or default_methods

    result: Dict[str, Dict] = {}
    rows = []

    for name, method in method_map.items():
        try:
            result[name] = identify_outliers(df, method=method)
        except Exception as exc:
            logger.warning("Method %s skipped: %s", name, exc)
            continue

    for method_name, cols in result.items():
        for col_name, state in cols.items():
            rows.append({
                "method":      method_name,
                "col":         col_name,
                "count":       state["count"],
                "percentage":  state["percentage"],
                "lower_bound": state.get("lower_bound"),
                "upper_bound": state.get("upper_bound"),
            })

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)
    metrics   = ["count", "percentage", "lower_bound", "upper_bound"]
    dfs_pivot = []

    for m in metrics:
        try:
            tmp = df_result.pivot(index="col", columns="method", values=m)
            tmp.columns = [f"{c}_{m}" for c in tmp.columns]
            dfs_pivot.append(tmp)
        except Exception:
            pass

    if not dfs_pivot:
        return df_result

    df_final = pd.concat(dfs_pivot, axis=1)

    ordered = [
        f"{m}_{metric}"
        for m in list(method_map.keys())
        for metric in metrics
        if f"{m}_{metric}" in df_final.columns
    ]

    df_final = df_final[[c for c in ordered if c in df_final.columns]]
    sort_col  = "zscore_count" if "zscore_count" in df_final.columns else df_final.columns[0]
    return df_final.sort_values(sort_col, ascending=False)


# Backward-compatible alias
respon_table_pandas = outlier_pivot_table


# ──────────────────────────────────────────────────────────────────────────────
# TREATMENT
# ──────────────────────────────────────────────────────────────────────────────


def treat_outliers(
    df: pd.DataFrame,
    outliers_dict: Dict[str, Dict[str, Any]],
    strategy: str = "winsorize",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply a treatment to detected outliers.

    Parameters
    ----------
    df            : source DataFrame
    outliers_dict : result of identify_outliers(return_mask=True)
    strategy      : 'winsorize' | 'drop' | 'median' | 'flag'
    verbose       : print a summary

    Returns
    -------
    pd.DataFrame — treated DataFrame
    """
    df_treated = df.copy()
    n_treated  = 0

    for col, info in outliers_dict.items():
        if info["count"] == 0:
            continue

        if strategy == "winsorize":
            lb = info.get("lower_bound")
            ub = info.get("upper_bound")
            if lb is not None and ub is not None:
                df_treated[col] = df_treated[col].clip(lower=lb, upper=ub)
                n_treated += 1

        elif strategy == "median":
            mask = info.get("mask")
            if mask is not None:
                df_treated.loc[mask, col] = df_treated[col].median()
                n_treated += 1

        elif strategy == "flag":
            mask = info.get("mask")
            if mask is not None:
                df_treated[f"outlier_{col}"] = mask.astype(int)
                n_treated += 1

        elif strategy == "drop":
            mask = info.get("mask")
            if mask is not None:
                df_treated = df_treated[~mask]
                n_treated += 1

    if verbose:
        print(f"\n Outlier treatment '{strategy}' applied to {n_treated} column(s).")
        if strategy == "drop":
            print(f"  Remaining rows: {len(df_treated):,}")

    if strategy == "drop":
        df_treated = df_treated.reset_index(drop=True)

    return df_treated


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────


def get_outlier_rows(
    df: pd.DataFrame,
    col: str,
    method: str = "zscore",
) -> pd.DataFrame:
    """Return the rows identified as outliers for a specific column."""
    outliers = identify_outliers(df, method=method, return_mask=True)
    if col not in outliers:
        raise ValueError(f"'{col}' not found in outlier results.")
    mask = outliers[col].get("mask")
    if mask is None:
        raise ValueError("Mask not available. Pass return_mask=True.")
    return df.loc[mask]


# Backward-compatible alias
outlier_m = get_outlier_rows


def combine_outlier_masks(
    df: pd.DataFrame,
    outliers_dict: Dict[str, Dict],
    verbose:bool = True
) -> pd.DataFrame:
    """
    Combine per-column outlier masks and return a comparison table.

    Returns a DataFrame with two rows per statistic: one for outlier rows,
    one for the full dataset, so you can compare profiles side by side.
    If no outliers are detected (common with uniform distributions), prints
    an explicit message and returns the full dataset describe() instead.
    """
    combined_mask = pd.Series(False, index=df.index)
    for col, info in outliers_dict.items():
        if "mask" not in info:
            raise ValueError(f"Mask missing for '{col}'. Pass return_mask=True.")
        combined_mask = combined_mask | info["mask"]

    n_outlier_rows = int(combined_mask.sum())
    n_total        = len(df)

    print(f"\n  Outlier rows (flagged in at least one column): "
          f"{n_outlier_rows:,} / {n_total:,} ({n_outlier_rows/n_total*100:.2f}%)")

    # Per-column summary regardless
    print(f"\n  {'Column':<25} {'Count':>6} {'Pct':>6}  {'Lower fence':>12} {'Upper fence':>12}")
    print("  " + "-" * 68)
    for col, info in outliers_dict.items():
        lb = f"{info['lower_bound']:.3f}" if info['lower_bound'] is not None else "N/A"
        ub = f"{info['upper_bound']:.3f}" if info['upper_bound'] is not None else "N/A"
        print(f"  {col:<25} {info['count']:>6} {info['percentage']:>5.1f}%  {lb:>12} {ub:>12}")

    if n_outlier_rows == 0 and  verbose:
        print("\n  No outlier rows detected across any column.")
        print("  This is expected for near-uniform distributions: the IQR fence")
        print("  extends beyond the data range, so no point falls outside it.")
        print("  Returning full dataset describe() for reference.\n")
        return df.select_dtypes(include=[np.number]).describe()

    outlier_desc = df[combined_mask].describe().add_suffix(" [outliers]")
    full_desc    = df.describe().add_suffix(" [all]")
    return pd.concat([outlier_desc, full_desc], axis=1).sort_index(axis=1)


# Backward-compatible alias
combine_mask_d = combine_outlier_masks


def plot_outlier_comparison(df: pd.DataFrame, col: str) -> None:
    """Visualize a column's outliers before and after winsorization."""
    outliers = identify_outliers(df, columns=[col], method="iqr",
                                  return_mask=True, verbose=False)
    if col not in outliers:
        print(f"  No outliers detected for '{col}'.")
        return

    info       = outliers[col]
    df_treated = treat_outliers(df, {col: info}, strategy="winsorize", verbose=False)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(x=df[col], ax=axes[0], color="salmon",
                flierprops={"marker": "o"})
    axes[0].set_title(f"{col} — Before\n({info['count']} IQR outliers)",
                      fontweight="bold")

    sns.boxplot(x=df_treated[col], ax=axes[1], color="steelblue")
    axes[1].set_title(f"{col} — After Winsorization", fontweight="bold")

    plt.suptitle(f"Outlier Treatment: {col}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def _print_outlier_summary(outliers: Dict, method: str) -> None:
    summary = pd.DataFrame([
        {"column": col, "n_outliers": v["count"], "percentage": v["percentage"]}
        for col, v in outliers.items()
    ]).sort_values("n_outliers", ascending=False)

    print(f"\n  Outlier Summary — method '{method}':")
    print(summary.to_string(index=False))
