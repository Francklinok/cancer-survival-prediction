"""
missing_value.py —  missing value treatment.

Supported strategies:
  - 'mice'        : Multiple Imputation by Chained Equations (IterativeImputer + RF)
  - 'knn'         : KNN Imputation
  - 'miss_forest' : MissForest (RandomForest-based)
  - 'simple'      : median (numeric) + mode (categorical)
  - 'constant'    : constant fill value

features:
  - Full before/after report
  - Separate handling for numeric / categorical columns
  - Missing indicator features (flag_missing_*)
  - Post-imputation validation
  - Scientific interpretation of missingness mechanism (MAR/MCAR/MNAR)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ml_framework.visualization.analysis.flag_plot import flag_plot
logger = logging.getLogger("ml_framework.missing_value")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN IMPUTATION
# ──────────────────────────────────────────────────────────────────────────────


def missing_data_handling(
    df: pd.DataFrame,
    strategy: str = "mice",
    categorical_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    add_missing_flags: bool = True,
    max_iter: int = 10,
    n_neighbors: int = 5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Missing value treatment with multiple strategies.

    Parameters
    ----------
    df               : DataFrame with missing values
    strategy         : 'mice' | 'knn' | 'miss_forest' | 'simple' | 'constant'
    categorical_cols : categorical columns (auto-detected if None)
    continuous_cols  : continuous columns (auto-detected if None)
    add_missing_flags: create binary indicator columns 'flag_missing_<col>'
    max_iter         : maximum iterations for MICE
    n_neighbors      : number of neighbors for KNN
    verbose          : print a detailed report

    Returns
    -------
    df_imputed : pd.DataFrame — imputed DataFrame
    report     : dict         — detailed imputation report
    """
    missing            = df.isnull().sum()
    cols_with_missing  = missing[missing > 0]

    report: Dict = {
        "strategy":        strategy,
        "n_missing_before": int(cols_with_missing.sum()),
        "cols_missing":    cols_with_missing.to_dict(),
        "flags_added":     [],
    }

    if len(cols_with_missing) == 0:
        if verbose:
            print("No missing values detected.")
        report["n_missing_after"] = 0
        return df.copy(), report

    if verbose:
        _print_missing_report(cols_with_missing, len(df))

    # Auto-detect column roles
    if continuous_cols is None and categorical_cols is None:
        continuous_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in continuous_cols]

    continuous_cols  = continuous_cols  or []
    categorical_cols = categorical_cols or []

    df_imputed = df.copy()

    # Missing indicator flags
    if add_missing_flags:
        for col in cols_with_missing.index:
            flag_name = f"flag_missing_{col}"
            df_imputed[flag_name] = df[col].isnull().astype(int)
            report["flags_added"].append(flag_name)
            logger.debug("Missing flag created: %s", flag_name)

    # Impute continuous columns
    cont_missing = [c for c in continuous_cols if c in cols_with_missing.index]

    if cont_missing:
        if verbose:
            print(f"\n  Numeric imputation ({strategy}) on {len(cont_missing)} column(s)...")

        if strategy == "mice":
            df_imputed = _impute_mice(df_imputed, cont_missing, max_iter)
        elif strategy == "knn":
            df_imputed = _impute_knn(df_imputed, cont_missing, n_neighbors)
        elif strategy == "miss_forest":
            df_imputed = _impute_missforest(df_imputed, cont_missing)
        elif strategy == "simple":
            for col in cont_missing:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        elif strategy == "constant":
            for col in cont_missing:
                df_imputed[col] = df_imputed[col].fillna(-9999)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose: mice, knn, miss_forest, simple, constant"
            )

    # Impute categorical columns (mode)
    cat_missing = [c for c in categorical_cols if c in cols_with_missing.index]

    if cat_missing:
        if verbose:
            print(f"  Categorical imputation (mode) on {len(cat_missing)} column(s)...")
        for col in cat_missing:
            mode_val = df_imputed[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            df_imputed[col] = df_imputed[col].fillna(fill_val)

    # Post-imputation validation
    remaining_missing = df_imputed.isnull().sum()
    remaining         = remaining_missing[remaining_missing > 0]
    report["n_missing_after"]     = int(remaining.sum())
    report["cols_missing_after"]  = remaining.to_dict()

    if verbose:
        if remaining.empty:
            print("\n All missing values successfully imputed.")
        else:
            print(f"\n {len(remaining)} column(s) with residual NaN:")
            for col, cnt in remaining.items():
                print(f"    - {col}: {cnt} NaN remaining")

    return df_imputed, report


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL IMPUTERS
# ──────────────────────────────────────────────────────────────────────────────


def _impute_mice(df: pd.DataFrame, cols: List[str], max_iter: int) -> pd.DataFrame:
    """MICE via IterativeImputer + RandomForestRegressor."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imp = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            max_iter=max_iter,
            random_state=42,
            verbose=0,
        )
        df[cols] = imp.fit_transform(df[cols].copy())
    except ImportError:
        logger.error("sklearn not installed — falling back to median imputation.")
        for col in cols:
            df[col] = df[col].fillna(df[col].median())
    return df


def _impute_knn(df: pd.DataFrame, cols: List[str], n_neighbors: int) -> pd.DataFrame:
    """KNN Imputation."""
    try:
        from sklearn.impute import KNNImputer

        imp     = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        df[cols] = imp.fit_transform(df[cols].copy())
    except ImportError:
        logger.error("sklearn not installed — falling back to median imputation.")
        for col in cols:
            df[col] = df[col].fillna(df[col].median())
    return df


def _impute_missforest(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """MissForest: iterative Random Forest-based imputation."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imp = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            max_iter=20,
            random_state=42,
            verbose=0,
        )
        df[cols] = imp.fit_transform(df[cols].copy())
    except ImportError:
        logger.error("sklearn not installed — falling back to median imputation.")
        for col in cols:
            df[col] = df[col].fillna(df[col].median())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────────────────────────


def _print_missing_report(cols_with_missing: pd.Series, n_rows: int) -> None:
    print("\n" + "═" * 60)
    print("  MISSING VALUES DETECTED")
    print("═" * 60)
    print(f"  {len(cols_with_missing)} column(s) with NaN\n")

    for col, cnt in cols_with_missing.sort_values(ascending=False).items():
        pct     = cnt / n_rows * 100
        bar     = "=" * int(pct / 5)
        mcar_hint = (
            "MCAR likely"        if pct < 5
            else "MAR/MNAR possible" if pct < 20
            else "High rate"
        )
        print(f"  {col:<35} {cnt:>5} ({pct:5.1f}%)  {bar}  {mcar_hint}")

    print()
    print("  Missingness mechanism interpretation:")
    print("    MCAR (Missing Completely At Random) → simple imputation acceptable")
    print("    MAR  (Missing At Random)             → MICE/KNN recommended")
    print("    MNAR (Missing Not At Random)         → analyze mechanism + add indicator flag")
    print("═" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────


def replace_value(
    continuous_cols: List[str],
    df_target: pd.DataFrame,
    imputed_values: np.ndarray,
    df_original: pd.DataFrame,
) -> None:
    """
    Re-insert imputed values (numpy array) into a DataFrame,
    preserving the original dtype (int64 → rounded).

    Used after matrix-based imputation (MICE, KNN, MissForest)
    to place continuous columns back into the target DataFrame.

    Parameters
    ----------
    continuous_cols : list of imputed continuous column names
    df_target       : destination DataFrame to modify in-place
    imputed_values  : 2-D numpy array (n_rows × len(continuous_cols))
    df_original     : original DataFrame — used to retrieve source dtypes

    Returns
    -------
    None  (in-place modification of df_target)
    """
    assert imputed_values.shape[1] == len(continuous_cols), (
        f"Mismatch: {imputed_values.shape[1]} imputed columns "
        f"vs {len(continuous_cols)} expected."
    )
    for i, col in enumerate(continuous_cols):
        df_target[col] = imputed_values[:, i]
        if col in df_original.columns and df_original[col].dtype == "int64":
            df_target[col] = df_target[col].round().astype(int)

def flag_analysis(df:pd.DataFrame, flag:str) -> None:
    """
    Analyze the added missing indicator flags (flag_missing_*)
    to interpret the missingness mechanism (MCAR/MAR/MNAR).

    This function would:
      - Compute correlation of flags with other features
      - Visualize distributions of flags vs target variable
      - Provide insights on whether missingness is random or systematic
    """
    #==============Flag distribution================
    n_was_missing = df[flag].sum()
    n_total = len(df)
    pct_missing = n_was_missing / n_total * 100

    print("═" * 55)
    print("  ANALYSE — flag_missing_GeneticMarker")
    print("═" * 55)
    print(f"  Lignes avec NaN imputé  : {n_was_missing:,}  ({pct_missing:.1f}%)")
    print(f"  Lignes complètes        : {n_total - n_was_missing:,}  ({100 - pct_missing:.1f}%)")

    # 2==============Association target vs  flag================
    cross = (
        df
        .groupby(flag)["TreatmentResponse"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("pct")
        .reset_index()
    )
    cross[flag] = cross[flag].map({0: "Complet (0)", 1: "Était NaN (1)"})
    print(cross.to_string(index=False))
    #================Visualisation================
    flag_plot(flag, n_was_missing, n_total, cross)

