"""
data_profiling.py 

Covers:
  - Column identification and categorization (continuous, discrete, binary,
    ordinal, categorical, datetime, identifier)
  - Numeric and categorical descriptive statistics
  - Extended per-column quality report (12 axes):
      1.  Type detection & possible conversion
      2.  Missing values (NaN)
      3.  Duplicates (rows and columns)
      4.  Zero variance / constant columns
      5.  Cardinality (ratio, high-cardinality, possible reduction)
      6.  Inconsistent values
      7.  Rare categories
      8.  Invalid values
      9.  String cleaning
     10.  Date cleaning
     11.  Data leakage detection
     12.  Skewness / Kurtosis / IQR outliers
  - Global quality score (0–100)
  - Formatted overview for notebook / terminal use
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.data_profiling")

# ── Global constants ──────────────────────────────────────────────────────────

_DEFAULT_ORDINAL_COLS: List[str] = [
    "Cancer_Stage", "Obesity_BMI", "Physical_Activity",
    "Diet_Risk", "Economic_Classification", "Healthcare_Access",
    "SmokingStatus", "Stage", "TreatmentResponse",
    "Survival_Category", "Age_Group", "BMI_Category",
    "Tumor_Size_Category",
]

_HIGH_CARDINALITY_RATIO: float = 0.9
_ID_PATTERN_KEYWORDS: List[str] = ["id", "code", "key", "uuid", "ref", "num", "no"]
_RARE_CATEGORY_THRESHOLD: float = 0.01   # < 1% → rare
_STRING_NOISE_PATTERN = re.compile(r"^\s+|\s+$|[\t\n\r]")

# Common invalid value patterns (medical domain + generic)
_INVALID_SENTINELS = {-9999, -999, -99, -1, 9999, 999, 99999}
_INVALID_STRINGS   = {"na", "n/a", "nan", "null", "none", "?", "unknown",
                      "missing", "not available", "-", "--", ""}

# Target keywords that trigger a leakage alert
_LEAKAGE_TARGET_KEYWORDS: List[str] = [
    "target", "label", "outcome", "response", "result",
    "survived", "recurrence", "death", "status",
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. COLUMN ROLE IDENTIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def check_columns_types(
    df: pd.DataFrame,
    discrete_threshold: int = 10,
    known_ordinal_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Identify and categorize all columns by their statistical role.

    Returns
    -------
    dict with keys:
        numerical_cols, continuous_cols, discrete_cols,
        categorical_cols, binary_cols, ordinal_cols,
        datetime_cols, id_cols, high_cardinality_cols
    """
    known_ordinal = set(known_ordinal_cols or _DEFAULT_ORDINAL_COLS)
    n = len(df)

    numerical_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    continuous_cols: List[str] = [
        c for c in numerical_cols if df[c].nunique(dropna=True) > discrete_threshold
    ]
    discrete_cols: List[str] = [c for c in numerical_cols if c not in continuous_cols]

    categorical_cols: List[str] = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    binary_cols: List[str] = [
        c for c in df.columns if df[c].nunique(dropna=True) == 2
    ]

    ordinal_cols: List[str] = [c for c in known_ordinal if c in df.columns]

    # ── Datetime: native dtype OR parse attempt ───────────────────────────────
    datetime_cols: List[str] = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    for col in categorical_cols:
        if col in datetime_cols:
            continue
        sample = df[col].dropna().head(200)
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                datetime_cols.append(col)
        except Exception:
            pass

    # ── ID columns ────────────────────────────────────────────────────────────
    id_cols: List[str] = []
    for col in df.columns:
        col_lower = col.lower()
        is_id_name = any(kw in col_lower for kw in _ID_PATTERN_KEYWORDS)
        ratio = df[col].nunique(dropna=True) / n if n > 0 else 0
        if is_id_name and ratio > 0.8:
            id_cols.append(col)
        elif ratio >= 1.0 and col not in numerical_cols:
            id_cols.append(col)

    # ── High-cardinality categoricals ────────────────────────────────────────
    high_cardinality_cols: List[str] = [
        c for c in categorical_cols
        if c not in id_cols
        and (df[c].nunique(dropna=True) / n if n > 0 else 0) > _HIGH_CARDINALITY_RATIO
    ]

    result = {
        "numerical_cols":        numerical_cols,
        "continuous_cols":       continuous_cols,
        "discrete_cols":         discrete_cols,
        "categorical_cols":      categorical_cols,
        "binary_cols":           binary_cols,
        "ordinal_cols":          ordinal_cols,
        "datetime_cols":         datetime_cols,
        "id_cols":               id_cols,
        "high_cardinality_cols": high_cardinality_cols,
    }

    logger.info(
        "Columns identified — numeric: %d | continuous: %d | discrete: %d "
        "| categorical: %d | binary: %d | ordinal: %d | datetime: %d "
        "| id: %d | high-cardinality: %d",
        len(numerical_cols), len(continuous_cols), len(discrete_cols),
        len(categorical_cols), len(binary_cols), len(ordinal_cols),
        len(datetime_cols), len(id_cols), len(high_cardinality_cols),
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def dataset_overview(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descriptive statistics optimised for **exploration and visualisation**.

    Distinct role from ``full_quality_report``:

    * ``dataset_overview``       — *wide* summary per dtype group; full percentile
      ladder (1 %, 5 %, 25 %, 50 %, 75 %, 95 %, 99 %); intended for quick
      inspection and notebook display.
    * ``full_quality_report``    — *tall* audit table (one row per column); 12
      quality axes (leakage, invalids, rare cats, …); reuses the stats computed
      here via the ``num_stats`` cache to avoid double computation.

    Parameters
    ----------
    df      : DataFrame to profile
    verbose : if True, print the formatted tables

    Returns
    -------
    num_summary : pd.DataFrame
        ``describe().T`` enriched with skewness, excess kurtosis, missing_pct.
        Index = column names.  Columns include the full percentile set.
    cat_summary : pd.DataFrame
        ``describe().T`` for object/category columns + missing_pct.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Parameter 'df' must be a valid pandas DataFrame.")

    num_df = df.select_dtypes(include=[np.number])
    num_summary = pd.DataFrame(
        num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    )
    if not num_summary.empty:
        num_summary["skewness"]    = num_df.skew()
        num_summary["kurtosis"]    = num_df.kurtosis()
        num_summary["missing_pct"] = (num_df.isnull().mean() * 100).round(2)

    cat_df = df.select_dtypes(include=["object", "category"])
    cat_summary = pd.DataFrame(cat_df.describe().T)
    if not cat_summary.empty:
        cat_summary["missing_pct"] = (cat_df.isnull().mean() * 100).round(2)

    if verbose:
        print("\n" + "═" * 70)
        print("  NUMERIC STATISTICS")
        print("═" * 70)
        print(num_summary.to_string())
        if not cat_summary.empty:
            print("\n" + "═" * 70)
            print("  CATEGORICAL STATISTICS")
            print("═" * 70)
            print(cat_summary.to_string())

    return num_summary, cat_summary


def _build_num_stats_cache(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute per-column numeric statistics **once** and return a dict keyed by
    column name.

    Used internally by ``full_quality_report`` so that it never recomputes
    values already available from ``dataset_overview``.

    Keys per column
    ---------------
    mean, std, min, max, skewness, kurtosis,
    has_negative, has_zero, n_outliers_iqr, outlier_pct, zero_variance
    """
    cache: Dict[str, Dict] = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 2:
            cache[col] = {"zero_variance": True}
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        if iqr > 0:
            n_out     = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            out_pct   = round(n_out / len(s) * 100, 2)
        else:
            n_out, out_pct = 0, 0.0
        cache[col] = {
            "mean":           round(float(s.mean()),     4),
            "std":            round(float(s.std()),      4),
            "min":            round(float(s.min()),      4),
            "max":            round(float(s.max()),      4),
            "skewness":       round(float(s.skew()),     4),
            "kurtosis":       round(float(s.kurtosis()), 4),
            "has_negative":   bool((s < 0).any()),
            "has_zero":       bool((s == 0).any()),
            "n_outliers_iqr": n_out,
            "outlier_pct":    out_pct,
            "zero_variance":  bool(s.std() < 1e-8),
        }
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL DETECTORS 
# ──────────────────────────────────────────────────────────────────────────────

def _detect_type_conversion(series: pd.Series) -> Optional[str]:
    """
    Detect if an object column can be converted to a more precise type.
    Returns the suggested target type or None.
    """
    if not pd.api.types.is_object_dtype(series):
        return None
    sample = series.dropna().head(500)
    if sample.empty:
        return None

    # Numeric attempt
    try:
        converted = pd.to_numeric(sample, errors="coerce")
        if converted.notna().mean() > 0.95:
            if (converted == converted.round()).all():
                return "int64"
            return "float64"
    except Exception:
        pass

    # Datetime attempt
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        if parsed.notna().mean() > 0.80:
            return "datetime64"
    except Exception:
        pass

    # Bool attempt
    bool_vals = {"true", "false", "yes", "no", "1", "0", "oui", "non"}
    if set(sample.str.lower().unique()).issubset(bool_vals):
        return "bool"

    # Category attempt (low cardinality)
    if series.nunique(dropna=True) / max(len(series), 1) < 0.05:
        return "category"

    return None


def _detect_inconsistent_values(series: pd.Series) -> Dict:
    """
    Detect inconsistent values in a categorical column:
    - Label duplicates (case, spaces)
    - Near-orthographic variants
    """
    result = {"has_inconsistency": False, "examples": []}
    if not pd.api.types.is_object_dtype(series):
        return result

    raw_vals   = series.dropna().unique().tolist()
    normalized = [str(v).strip().lower() for v in raw_vals]
    seen: Dict[str, str] = {}
    conflicts  = []

    for raw, norm in zip(raw_vals, normalized):
        if norm in seen and seen[norm] != str(raw):
            conflicts.append(f"'{seen[norm]}' vs '{raw}'")
        else:
            seen[norm] = str(raw)

    if conflicts:
        result["has_inconsistency"] = True
        result["examples"] = conflicts[:5]

    return result


def _detect_rare_categories(series: pd.Series, threshold: float = _RARE_CATEGORY_THRESHOLD) -> Dict:
    """
    Detect categories whose frequency is below the threshold.
    """
    result = {"has_rare": False, "rare_values": [], "rare_pct": 0.0}
    if not pd.api.types.is_object_dtype(series) and \
       not pd.api.types.is_categorical_dtype(series):
        return result

    n = len(series.dropna())
    if n == 0:
        return result

    vc = series.value_counts(normalize=True, dropna=True)
    rare = vc[vc < threshold]
    if not rare.empty:
        result["has_rare"]    = True
        result["rare_values"] = rare.index.tolist()[:10]
        result["rare_pct"]    = round(float(rare.sum() * 100), 2)

    return result


def _detect_invalid_values(series: pd.Series) -> Dict:
    """
    Detect invalid values:
    - Numeric sentinel values (-9999, 999, etc.)
    - Strings representing missing values ('N/A', '?', 'null', etc.)
    - Unlikely negative values (Age, BMI, etc.)
    """
    result = {"has_invalid": False, "invalid_count": 0, "details": []}

    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        sentinels_found = s[s.isin(_INVALID_SENTINELS)]
        if not sentinels_found.empty:
            result["has_invalid"]    = True
            result["invalid_count"] += len(sentinels_found)
            result["details"].append(
                f"Numeric sentinels: {sentinels_found.unique().tolist()[:5]}"
            )

    elif pd.api.types.is_object_dtype(series):
        s_lower = series.dropna().str.strip().str.lower()
        mask    = s_lower.isin(_INVALID_STRINGS)
        count   = int(mask.sum())
        if count > 0:
            result["has_invalid"]    = True
            result["invalid_count"] += count
            result["details"].append(
                f"Invalid strings masking NaN: {count} value(s)"
            )

    return result


def _detect_string_issues(series: pd.Series) -> Dict:
    """
    Detect string cleaning issues:
    - Leading/trailing spaces
    - Control characters (\\t, \\n, \\r)
    - Mixed case (e.g. 'Male' vs 'male' vs 'MALE')
    - Abnormal length
    """
    result = {
        "has_leading_trailing_spaces": False,
        "has_control_chars": False,
        "has_mixed_case": False,
        "n_dirty": 0,
    }
    if not pd.api.types.is_object_dtype(series):
        return result

    s       = series.dropna().astype(str)
    n_dirty = 0

    # Leading/trailing spaces
    has_spaces = s.str.contains(r"^\s+|\s+$", regex=True)
    if has_spaces.any():
        result["has_leading_trailing_spaces"] = True
        n_dirty += int(has_spaces.sum())

    # Control characters \t \n \r
    has_ctrl = s.apply(lambda x: bool(_STRING_NOISE_PATTERN.search(x)))
    if has_ctrl.any():
        result["has_control_chars"] = True
        n_dirty += int(has_ctrl.sum())

    # Mixed case (same normalized value in multiple cases)
    normalized = s.str.strip().str.lower()
    for _, group in s.groupby(normalized):
        if len(group.unique()) > 1:
            result["has_mixed_case"] = True
            break

    result["n_dirty"] = n_dirty
    return result


def _detect_date_issues(series: pd.Series) -> Dict:
    """
    Detect date column issues:
    - Unlikely future dates
    - Too-old dates (before 1900)
    - Inconsistent formats (mixed formats)
    """
    result = {
        "has_future_dates": False,
        "has_ancient_dates": False,
        "n_future": 0,
        "n_ancient": 0,
    }

    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            s = series.dropna()
        else:
            s = pd.to_datetime(series.dropna(), infer_datetime_format=True, errors="coerce").dropna()

        if s.empty:
            return result

        now    = pd.Timestamp.now()
        future = s[s > now]
        old    = s[s.dt.year < 1900]

        result["has_future_dates"]  = not future.empty
        result["n_future"]          = len(future)
        result["has_ancient_dates"] = not old.empty
        result["n_ancient"]         = len(old)

    except Exception:
        pass

    return result


def _detect_leakage(
    series: pd.Series,
    col_name: str,
    target_col: Optional[str] = None,
) -> Dict:
    """
    Detect data leakage signals:
    - Column name contains a target keyword
    - Derived/post-processing column prefix
    - Numeric column with all values in [0,1] → potential predicted probabilities
    """
    result = {"leakage_risk": "none", "reason": ""}

    col_lower = col_name.lower()

    # 1. Suspicious name (target keyword, but not the target itself)
    if target_col and col_name != target_col:
        if any(kw in col_lower for kw in _LEAKAGE_TARGET_KEYWORDS):
            result["leakage_risk"] = "high"
            result["reason"] = (
                f"Column name '{col_name}' contains a target keyword "
                f"({[kw for kw in _LEAKAGE_TARGET_KEYWORDS if kw in col_lower]})"
            )
            return result

    # 2. Derived / post-imputation flag (common prefixes)
    derived_prefixes = ["flag_", "target_", "leak_", "encoded_", "pred_", "proba_"]
    if any(col_lower.startswith(p) for p in derived_prefixes):
        result["leakage_risk"] = "medium"
        result["reason"] = f"Suspicious prefix indicating a post-processing column: '{col_name}'"
        return result

    # 3. Values in [0, 1] with many unique values → likely predicted probabilities
    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        if len(s) > 10 and s.between(0, 1).all() and s.nunique() > 10:
            result["leakage_risk"] = "medium"
            result["reason"] = "Numeric column with values in [0,1] → potential predicted probabilities"

    return result


def _detect_cardinality_reduction(series: pd.Series) -> Dict:
    """
    Indicate if cardinality can be reduced:
    - Too many categories (> 20) → grouping recommended
    - Numeric column with few unique values → may become discrete/categorical
    """
    result = {"can_reduce": False, "suggestion": ""}
    n_unique = series.nunique(dropna=True)

    if pd.api.types.is_object_dtype(series):
        if n_unique > 20:
            result["can_reduce"]  = True
            result["suggestion"]  = f"{n_unique} categories → grouping ('Other') recommended"
        elif n_unique > 10:
            result["can_reduce"]  = True
            result["suggestion"]  = f"{n_unique} categories → consider target/ordinal encoding"

    elif pd.api.types.is_numeric_dtype(series):
        if 2 < n_unique <= 10:
            result["can_reduce"]  = True
            result["suggestion"]  = f"Numeric with {n_unique} values → can be treated as discrete"

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. COMPLETE QUALITY REPORT
# ──────────────────────────────────────────────────────────────────────────────

def full_quality_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Production-grade per-column quality audit (12 axes).

    Distinct role from ``dataset_overview``:

    * ``dataset_overview``    — wide exploration table with full percentile
      ladder; best for notebook display and quick inspection.
    * ``full_quality_report`` — tall audit table (one row per column) covering
      12 quality axes: type conversion, missing values, duplicates, zero
      variance, cardinality, inconsistencies, rare categories, invalid values,
      string issues, date issues, leakage, and numeric stats.  Numeric stats
      are computed **once** via ``_build_num_stats_cache`` and reused here
      — no double computation with ``dataset_overview``.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str, optional — target column (sharpens leakage detection)

    Returns
    -------
    pd.DataFrame — one row per column, indexed by column name.
    Attributes:
        .attrs["n_duplicate_rows"]   : int
        .attrs["target_col"]         : str | None
    """
    n_rows     = len(df)
    type_info  = check_columns_types(df)
    id_col_set = set(type_info["id_cols"])
    dt_col_set = set(type_info["datetime_cols"])
    hc_col_set = set(type_info["high_cardinality_cols"])
    all_cols   = df.columns.tolist()

    # Compute numeric stats once — reused for every numeric column below
    num_cache: Dict[str, Dict] = _build_num_stats_cache(df)

    # ── Duplicate columns (same content) ─────────────────────────────────────
    duplicate_col_names: set = set()
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            try:
                if df[all_cols[i]].equals(df[all_cols[j]]):
                    duplicate_col_names.add(all_cols[j])
            except Exception:
                pass

    n_duplicate_rows = int(df.duplicated().sum())

    rows = []
    for col in all_cols:
        series     = df[col]
        n_missing  = int(series.isnull().sum())
        n_unique   = int(series.nunique(dropna=True))
        dtype      = str(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_object  = pd.api.types.is_object_dtype(series)
        is_dt      = col in dt_col_set

        # ── Initialization ────────────────────────────────────────────────────
        row: Dict = {
            # General
            "dtype":                    dtype,
            "n_total":                  n_rows,
            "n_unique":                 n_unique,
            "missing_count":            n_missing,
            "missing_pct":              round(n_missing / n_rows * 100, 4) if n_rows else 0,
            "is_constant":              n_unique <= 1,
            "zero_variance":            False,
            "is_duplicate_col":         col in duplicate_col_names,
            "is_id_col":                col in id_col_set,
            "is_datetime":              is_dt,
            # 1. Type conversion
            "suggested_type":           None,
            # 5. Cardinality
            "cardinality_ratio":        round(n_unique / n_rows, 4) if n_rows else 0,
            "is_high_cardinality":      col in hc_col_set,
            "cardinality_reduction":    "",
            # 6. Inconsistencies
            "has_inconsistent_values":  False,
            "inconsistent_examples":    "",
            # 7. Rare categories
            "has_rare_categories":      False,
            "rare_categories":          "",
            "rare_pct":                 0.0,
            # 8. Invalid values
            "has_invalid_values":       False,
            "invalid_count":            0,
            "invalid_details":          "",
            # 9. String cleaning
            "needs_string_cleaning":    False,
            "string_issues":            "",
            # 10. Date cleaning
            "has_date_issues":          False,
            "date_issue_details":       "",
            # 11. Leakage
            "leakage_risk":             "none",
            "leakage_reason":           "",
            # 12. Numeric stats
            "mean":                     None,
            "std":                      None,
            "min":                      None,
            "max":                      None,
            "skewness":                 None,
            "kurtosis":                 None,
            "has_negative":             False,
            "has_zero":                 False,
            "n_outliers_iqr":           None,
            "outlier_pct":              None,
            # Categorical
            "top_value":                None,
            "top_freq_pct":             None,
        }

        # ── 1. Type conversion ────────────────────────────────────────────────
        row["suggested_type"] = _detect_type_conversion(series)

        # ── 4 + 12. Zero variance & numeric stats (read from cache) ─────────────
        if is_numeric:
            stats = num_cache.get(col, {})
            row["zero_variance"]  = stats.get("zero_variance",  True)
            row["mean"]           = stats.get("mean")
            row["std"]            = stats.get("std")
            row["min"]            = stats.get("min")
            row["max"]            = stats.get("max")
            row["skewness"]       = stats.get("skewness")
            row["kurtosis"]       = stats.get("kurtosis")
            row["has_negative"]   = stats.get("has_negative",   False)
            row["has_zero"]       = stats.get("has_zero",       False)
            row["n_outliers_iqr"] = stats.get("n_outliers_iqr")
            row["outlier_pct"]    = stats.get("outlier_pct")

        else:
            # Top categorical value
            vc = series.value_counts(dropna=True)
            if not vc.empty:
                row["top_value"]    = str(vc.index[0])
                row["top_freq_pct"] = round(vc.iloc[0] / n_rows * 100, 2)

        # ── 5. Cardinality reduction ──────────────────────────────────────────
        card_info = _detect_cardinality_reduction(series)
        if card_info["can_reduce"]:
            row["cardinality_reduction"] = card_info["suggestion"]

        # ── 6. Inconsistent values ────────────────────────────────────────────
        if is_object:
            incons = _detect_inconsistent_values(series)
            row["has_inconsistent_values"] = incons["has_inconsistency"]
            row["inconsistent_examples"]   = " | ".join(incons["examples"])

        # ── 7. Rare categories ────────────────────────────────────────────────
        rare = _detect_rare_categories(series)
        row["has_rare_categories"] = rare["has_rare"]
        row["rare_categories"]     = ", ".join(str(v) for v in rare["rare_values"])
        row["rare_pct"]            = rare["rare_pct"]

        # ── 8. Invalid values ─────────────────────────────────────────────────
        inv = _detect_invalid_values(series)
        row["has_invalid_values"] = inv["has_invalid"]
        row["invalid_count"]      = inv["invalid_count"]
        row["invalid_details"]    = " | ".join(inv["details"])

        # ── 9. String cleaning ────────────────────────────────────────────────
        if is_object:
            str_issues = _detect_string_issues(series)
            issues_list = []
            if str_issues["has_leading_trailing_spaces"]:
                issues_list.append("leading/trailing spaces")
            if str_issues["has_control_chars"]:
                issues_list.append("control chars (\\t\\n)")
            if str_issues["has_mixed_case"]:
                issues_list.append("mixed case")
            row["needs_string_cleaning"] = bool(issues_list)
            row["string_issues"]         = " | ".join(issues_list)

        # ── 10. Date cleaning ─────────────────────────────────────────────────
        if is_dt or is_object:
            date_info = _detect_date_issues(series)
            date_details = []
            if date_info["has_future_dates"]:
                date_details.append(f"{date_info['n_future']} future dates")
            if date_info["has_ancient_dates"]:
                date_details.append(f"{date_info['n_ancient']} dates before 1900")
            if date_details:
                row["has_date_issues"]    = True
                row["date_issue_details"] = " | ".join(date_details)

        # ── 11. Leakage detection ─────────────────────────────────────────────
        leak = _detect_leakage(series, col, target_col)
        row["leakage_risk"]   = leak["leakage_risk"]
        row["leakage_reason"] = leak["reason"]

        rows.append({"column": col, **row})

    report_df = pd.DataFrame(rows).set_index("column")

    # ── Structured alerts ─────────────────────────────────────────────────────
    _log_quality_alerts(report_df, n_duplicate_rows)

    report_df.attrs["n_duplicate_rows"] = n_duplicate_rows
    report_df.attrs["target_col"]       = target_col

    return report_df


def _log_quality_alerts(report_df: pd.DataFrame, n_duplicate_rows: int) -> None:
    """Log detected quality alerts."""
    issues = []

    checks = [
        (n_duplicate_rows,                                 f"{n_duplicate_rows:,} duplicate row(s)"),
        (int((report_df["missing_pct"] > 0).sum()),        f"{int((report_df['missing_pct'] > 0).sum())} column(s) with NaN"),
        (int(report_df["is_constant"].sum()),              f"{int(report_df['is_constant'].sum())} constant column(s)"),
        (int(report_df["zero_variance"].sum()),            f"{int(report_df['zero_variance'].sum())} zero-variance column(s)"),
        (int(report_df["is_duplicate_col"].sum()),         f"{int(report_df['is_duplicate_col'].sum())} duplicate column(s)"),
        (int(report_df["has_inconsistent_values"].sum()),  f"{int(report_df['has_inconsistent_values'].sum())} inconsistent column(s)"),
        (int(report_df["has_rare_categories"].sum()),      f"{int(report_df['has_rare_categories'].sum())} column(s) with rare categories"),
        (int(report_df["has_invalid_values"].sum()),       f"{int(report_df['has_invalid_values'].sum())} column(s) with invalid values"),
        (int(report_df["needs_string_cleaning"].sum()),    f"{int(report_df['needs_string_cleaning'].sum())} column(s) needing string cleanup"),
        (int(report_df["has_date_issues"].sum()),          f"{int(report_df['has_date_issues'].sum())} column(s) with date issues"),
        ((report_df["leakage_risk"] != "none").sum(),      f"{(report_df['leakage_risk'] != 'none').sum()} column(s) with leakage risk"),
        (int(report_df["is_high_cardinality"].sum()),      f"{int(report_df['is_high_cardinality'].sum())} high-cardinality column(s)"),
    ]

    for count, msg in checks:
        if count:
            issues.append(msg)

    if issues:
        logger.warning("Quality issues detected → %s", " | ".join(issues))
    else:
        logger.info("No quality issues detected ✓")


# ──────────────────────────────────────────────────────────────────────────────
# 4. GLOBAL QUALITY SCORE
# ──────────────────────────────────────────────────────────────────────────────

def compute_quality_score(
    df: pd.DataFrame,
    report_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute a global dataset quality score (0 → 100).

    Penalties (total 100 pts):
      Average NaN             → −20 pts max
      Constant columns        → −10 pts max
      Zero variance           → −10 pts max
      Duplicate rows          → −10 pts max
      Duplicate columns       → −5  pts max
      High cardinality        → −5  pts max
      Outliers > 5%           → −5  pts max
      Invalid values          → −10 pts max
      Inconsistent values     → −5  pts max
      Rare categories         → −5  pts max
      Dirty strings           → −5  pts max
      Leakage                 → −10 pts max
    """
    if report_df is None:
        report_df = full_quality_report(df)

    n_rows = len(df)
    n_cols = len(df.columns)
    detail: Dict[str, float] = {}

    def _pen(value: float, max_pen: float) -> float:
        return min(max_pen, round(value, 2))

    # NaN
    avg_missing = float(report_df["missing_pct"].mean())
    detail["nan_penalty"]                = _pen(avg_missing * 2.0, 20.0)

    # Constants
    detail["constant_penalty"]           = _pen(report_df["is_constant"].sum() / n_cols * 100 * 2.0, 10.0)

    # Zero variance
    detail["zero_variance_penalty"]      = _pen(report_df["zero_variance"].sum() / n_cols * 100 * 2.0, 10.0)

    # Duplicate rows
    n_dup_rows = report_df.attrs.get("n_duplicate_rows", int(df.duplicated().sum()))
    detail["duplicate_rows_penalty"]     = _pen(n_dup_rows / n_rows * 100 * 2.0, 10.0)

    # Duplicate columns
    detail["duplicate_cols_penalty"]     = _pen(report_df["is_duplicate_col"].sum() / n_cols * 100, 5.0)

    # High cardinality
    detail["high_cardinality_penalty"]   = _pen(report_df["is_high_cardinality"].sum() / n_cols * 100 * 0.5, 5.0)

    # Outliers
    out_mean = float(report_df["outlier_pct"].dropna().mean()) if report_df["outlier_pct"].dropna().any() else 0.0
    detail["outlier_penalty"]            = _pen(max(0.0, (out_mean - 5.0)), 5.0)

    # Invalid values
    detail["invalid_values_penalty"]     = _pen(report_df["has_invalid_values"].sum() / n_cols * 100 * 2.0, 10.0)

    # Inconsistent values
    detail["inconsistent_values_penalty"] = _pen(report_df["has_inconsistent_values"].sum() / n_cols * 100, 5.0)

    # Rare categories
    detail["rare_categories_penalty"]    = _pen(report_df["has_rare_categories"].sum() / n_cols * 100 * 0.5, 5.0)

    # Dirty strings
    detail["string_cleaning_penalty"]    = _pen(report_df["needs_string_cleaning"].sum() / n_cols * 100 * 0.5, 5.0)

    # Leakage
    n_leak = int((report_df["leakage_risk"] == "high").sum() * 2 +
                 (report_df["leakage_risk"] == "medium").sum())
    detail["leakage_penalty"]            = _pen(n_leak * 5.0, 10.0)

    total_penalty = sum(detail.values())
    score = max(0, round(100 - total_penalty))

    grade = (
        "A — Excellent"    if score >= 90 else
        "B — Good"         if score >= 75 else
        "C — Acceptable"   if score >= 60 else
        "D — Needs work"   if score >= 40 else
        "F — Critical"
    )

    return {"score": score, "grade": grade, "detail": detail}


# ──────────────────────────────────────────────────────────────────────────────
# 5. SYNTHETIC DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

def print_profiling_summary(
    df: pd.DataFrame,
    report_df: Optional[pd.DataFrame] = None,
    quality: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Print a complete profiling summary and return a structured findings DataFrame.

    The function:
      1. Prints the full profiling banner (dimensions, types, score, penalties, alerts)
      2. Computes and prints a **Key Observations** table (kurtosis, class balance,
         missing values, outliers, IDs) — the facts that feed the conclusion
      3. Returns a ``pd.DataFrame`` with columns:
           | Category | Column(s) | Observation | Implication |
         ready to display directly in a notebook cell.

    Pass ``report_df`` and/or ``quality`` if they were already computed upstream
    (e.g. after calling ``full_quality_report`` and ``compute_quality_score`` in
    the notebook) to avoid redundant computation.

    Parameters
    ----------
    df        : pd.DataFrame — raw or working DataFrame to profile.
    report_df : pd.DataFrame, optional — result of ``full_quality_report(df)``.
                Computed internally when not provided.
    quality   : dict, optional — result of ``compute_quality_score(df, report_df)``.
                Computed internally when not provided.

    Returns
    -------
    pd.DataFrame
        One row per notable finding — suitable for ``display()`` or ``.style``.
    """
    n_rows, n_cols = df.shape
    col_roles   = check_columns_types(df)
    if report_df is None:
        report_df = full_quality_report(df)
    if quality is None:
        quality = compute_quality_score(df, report_df)

    n_dup_rows  = report_df.attrs.get("n_duplicate_rows", 0)
    n_nan_total = int(df.isnull().sum().sum())
    mem_mb      = df.memory_usage(deep=True).sum() / 1024 ** 2
    W = 64

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n╔" + "═" * W + "╗")
    print("║{:^{w}}║".format("  COMPLETE DATASET PROFILING  ", w=W))
    print("╠" + "═" * W + "╣")
    print(f"║  {'Dimensions':<40} {n_rows:,} × {n_cols} columns")
    print(f"║  {'Memory':<40} {mem_mb:.2f} MB")
    print(f"║  {'Duplicate rows':<40} {n_dup_rows:,}")
    print(f"║  {'Total missing values':<40} {n_nan_total:,}")

    print("╠" + "═" * W + "╣")
    for label, key in [
        ("Continuous columns",        "continuous_cols"),
        ("Discrete columns",          "discrete_cols"),
        ("Categorical columns",       "categorical_cols"),
        ("Binary columns",            "binary_cols"),
        ("Ordinal columns",           "ordinal_cols"),
        ("Datetime columns",          "datetime_cols"),
        ("Identifier columns",        "id_cols"),
        ("High-cardinality columns",  "high_cardinality_cols"),
    ]:
        print(f"║  {label:<40} {len(col_roles[key])}")

    score_bar = "[" + "█" * round(quality["score"] / 10) + "░" * (10 - round(quality["score"] / 10)) + "]"
    print("╠" + "═" * W + "╣")
    print(f"║  {'Global quality score':<40} {quality['score']}/100  {score_bar}")
    print(f"║  {'Grade':<40} {quality['grade']}")

    print("╠" + "═" * W + "╣")
    print("║  Penalties by axis:")
    for key, val in quality["detail"].items():
        label  = key.replace("_penalty", "").replace("_", " ").capitalize()
        bar    = "░" * int(val)
        marker = " ⚠️" if val > 0 else " ✓"
        print(f"║    {label:<32} {val:5.1f} pts  {bar}{marker}")

    sections = [
        ("NaN (top 5)",            "missing_pct",             lambda r: r > 0,       "pct"),
        ("Invalid values",         "has_invalid_values",       lambda r: r,           "flag"),
        ("Inconsistent values",    "has_inconsistent_values",  lambda r: r,           "flag"),
        ("Rare categories",        "has_rare_categories",      lambda r: r,           "flag"),
        ("Strings to clean",       "needs_string_cleaning",    lambda r: r,           "flag"),
        ("Leakage risk",           "leakage_risk",             lambda r: r != "none", "raw"),
        ("IDs to exclude from ML", "is_id_col",                lambda r: r,           "id"),
        ("Duplicate columns",      "is_duplicate_col",         lambda r: r,           "dup"),
    ]
    for title, col, condition, fmt_type in sections:
        flagged = report_df[report_df[col].apply(condition)]
        if flagged.empty:
            continue
        print("╠" + "═" * W + "╣")
        print(f"║  {title}:")
        for c_name, row_data in flagged.head(5).iterrows():
            val = row_data[col]
            if fmt_type == "pct":
                val_str = f"{val:.2f}%"
            elif fmt_type == "raw":
                val_str = f"[{val}]"
            elif fmt_type == "id":
                val_str = "→ exclude from ML"
            elif fmt_type == "dup":
                val_str = "→ drop"
            else:
                val_str = "⚠️"
            print(f"║    • {str(c_name):<30} {val_str}")

    print("╚" + "═" * W + "╝\n")

    # ── Key Observations (the facts that feed the conclusion) ─────────────────
    findings: List[Dict] = []

    # 1. Kurtosis on numeric columns
    num_cols = col_roles.get("continuous_cols", []) + col_roles.get("discrete_cols", [])
    num_cols = [c for c in num_cols if c in df.columns and c not in col_roles.get("id_cols", [])]
    if num_cols:
        kurt_vals  = df[num_cols].kurtosis()
        skew_vals  = df[num_cols].skew()
        flat_cols  = kurt_vals[kurt_vals < -0.8].index.tolist()   # uniform-like
        skewed_cols = skew_vals[skew_vals.abs() > 1.0].index.tolist()

        if flat_cols:
            flat_str = ", ".join(flat_cols)
            kurt_mean = kurt_vals[flat_cols].mean()
            findings.append({
                "Category":    "Distribution",
                "Column(s)":   flat_str,
                "Observation": f"Kurtosis ≈ {kurt_mean:.1f} (flat / uniform-like)",
                "Implication": "Uniform distribution → standard normalization less effective; "
                               "MinMax or no scaling preferred",
            })
        if skewed_cols:
            sk_str = ", ".join(skewed_cols)
            findings.append({
                "Category":    "Distribution",
                "Column(s)":   sk_str,
                "Observation": f"Skewness > 1.0 detected",
                "Implication": "Log / Box-Cox transform recommended before modeling",
            })
        if not flat_cols and not skewed_cols:
            findings.append({
                "Category":    "Distribution",
                "Column(s)":   ", ".join(num_cols),
                "Observation": "Skewness and kurtosis within normal range",
                "Implication": "StandardScaler / RobustScaler applicable",
            })

    # 2. Class balance for categorical / ordinal targets
    cat_cols = col_roles.get("categorical_cols", []) + col_roles.get("ordinal_cols", [])
    for col in cat_cols:
        if col not in df.columns:
            continue
        vc  = df[col].value_counts(normalize=True).mul(100).round(1)
        n_c = len(vc)
        if n_c < 2 or n_c > 10:
            continue
        max_pct = vc.max()
        min_pct = vc.min()
        ratio   = max_pct / max(min_pct, 0.1)
        dist_str = " / ".join([f"{v:.1f}%" for v in vc.values])
        if ratio < 1.5:
            findings.append({
                "Category":    "Class Balance",
                "Column(s)":   col,
                "Observation": f"Balanced: {dist_str}",
                "Implication": "No class imbalance — no resampling / class_weight needed",
            })
        elif ratio < 3.0:
            findings.append({
                "Category":    "Class Balance",
                "Column(s)":   col,
                "Observation": f"Slightly imbalanced: {dist_str}  (ratio {ratio:.1f}×)",
                "Implication": "Monitor metrics (F1-macro, AUC); class_weight='balanced' optional",
            })
        else:
            findings.append({
                "Category":    "Class Balance",
                "Column(s)":   col,
                "Observation": f"Imbalanced: {dist_str}  (ratio {ratio:.1f}×)",
                "Implication": "Use class_weight='balanced' or SMOTE; prefer F1-macro / AUC-PR",
            })

    # 3. Missing values
    nan_cols = df.isnull().mean().mul(100).round(2)
    nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        for c_name, pct in nan_cols.head(5).items():
            mech = "MCAR" if pct < 5 else ("MAR" if pct < 30 else "MNAR (likely)")
            findings.append({
                "Category":    "Missing Values",
                "Column(s)":   c_name,
                "Observation": f"{pct:.2f}% NaN",
                "Implication": f"Mechanism likely {mech} → "
                               + ("simple imputation" if pct < 5
                                  else "MICE / KNN" if pct < 30
                                  else "add binary flag + impute"),
            })
    else:
        findings.append({
            "Category":    "Missing Values",
            "Column(s)":   "—",
            "Observation": "No missing values",
            "Implication": "✓ No imputation needed",
        })

    # 4. Duplicate rows
    findings.append({
        "Category":    "Duplicates",
        "Column(s)":   "—",
        "Observation": f"{n_dup_rows} duplicate row(s)",
        "Implication": "✓ No deduplication needed" if n_dup_rows == 0
                       else f"Drop {n_dup_rows} duplicates before modeling",
    })

    # 5. ID / leakage columns
    id_cols = col_roles.get("id_cols", [])
    if id_cols:
        findings.append({
            "Category":    "Identifier / Leakage",
            "Column(s)":   ", ".join(id_cols),
            "Observation": "Identifier column(s) detected",
            "Implication": "Exclude from features before training",
        })

    # 6. Quality score
    findings.append({
        "Category":    "Quality Score",
        "Column(s)":   "Global",
        "Observation": f"{quality['score']}/100 — {quality['grade']}",
        "Implication": "Ready for preprocessing"
                       if quality["score"] >= 80
                       else "Address flagged issues before modeling",
    })

    # ── Print Key Observations table ──────────────────────────────────────────
    findings_df = pd.DataFrame(findings, columns=["Category", "Column(s)", "Observation", "Implication"])

    print("\n" + "═" * 80)
    print("  KEY OBSERVATIONS  (facts for the conclusion)")
    print("═" * 80)
    print(findings_df.to_string(index=False))
    print("═" * 80 + "\n")

    return findings_df


# ──────────────────────────────────────────────────────────────────────────────
# 6. BACKWARD COMPATIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def check_columns_types_legacy(df: pd.DataFrame):
    """
    Backward compatibility: returns (numerical_cols, continue_cols, discret_cols,
    categorical_cols, binary_cols, ordinals_cols).
    """
    result = check_columns_types(df)
    return (
        result["numerical_cols"],
        result["continuous_cols"],
        result["discrete_cols"],
        result["categorical_cols"],
        result["binary_cols"],
        result["ordinal_cols"],
    )
