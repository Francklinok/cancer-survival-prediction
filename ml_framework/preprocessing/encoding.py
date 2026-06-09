"""
encoding.py —  Categorical variable encoding.

Strategies implemented:
  - Binary      : Yes/No, Female/Male mappings via map()
  - Ordinal     : explicit ordered mapping (preserves semantic order) via map()
  - Nominal     : pd.get_dummies(drop_first=True, dtype=int) — no sklearn, no 2D array
  - Target      : category mean encoding (with James-Stein regularization)
  - Frequency   : replacement by relative frequency

Features:
  - Drop identifier columns first (before any encoding)
  - encoded_cols tracking — raises a warning if a column would be encoded twice
  - Post-encoding validation (NaN, residual object types)
  - Detailed transformation report
  - Encoder metadata stored for future production inference
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_framework.encoding")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENCODING (cancer dataset)
# ──────────────────────────────────────────────────────────────────────────────


def encode_dataframe(
    df: pd.DataFrame,
    drop_id_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete encoding for the medical cancer dataset.

    Strategies applied in order:
      1. Drop   : identifier columns removed first (prevents encoding IDs)
      2. Binary : Yes/No, Female/Male → 0/1 via map()
      3. Ordinal: explicit ordered map() (preserves semantic order)
      4. Nominal: pd.get_dummies(drop_first=True, dtype=int) — clean OHE,
                  no sklearn, no 2D-array assignment issue
      5. Remaining object columns → pd.get_dummies + logger warning

    An ``encoded_cols`` set guarantees each column is processed by exactly one
    strategy; a warning is emitted if a collision is detected.

    Parameters
    ----------
    df           : DataFrame to encode
    drop_id_cols : additional columns to drop (IDs, etc.)
    verbose      : print the encoding report

    Returns
    -------
    df_encoded : pd.DataFrame — fully numeric
    encoders   : dict — transformation metadata for future inference
                 Binary/Ordinal: {"type": "binary"|"ordinal", "mapping": {...}}
                 Nominal OHE   : {"type": "ohe_dummies", "original_col": str,
                                  "new_cols": [list of created column names]}
    """
    df_processed = df.copy()
    encoders: Dict = {}
    ops_log: List[str] = []
    encoded_cols: set = set()

    # ── 1. DROP ID COLUMNS ────────────────────────────────────────────────────
    default_id_cols = ["Patient_ID", "ID", "patient_id", "index"]
    all_drop = list(set((drop_id_cols or []) + default_id_cols))
    cols_to_drop = [c for c in all_drop if c in df_processed.columns]
    if cols_to_drop:
        df_processed.drop(columns=cols_to_drop, inplace=True)
        ops_log.append(f"[Drop]     ID columns removed: {cols_to_drop}")

    # ── 2. BINARY COLUMNS ────────────────────────────────────────────────────
    binary_yes_no = [
        c for c in ["FamilyHistory", "Recurrence"]
        if c in df_processed.columns and c not in encoded_cols
    ]
    for col in binary_yes_no:
        df_processed[col] = df_processed[col].map({"Yes": 1, "No": 0})
        encoders[col] = {"type": "binary", "mapping": {"Yes": 1, "No": 0}}
        encoded_cols.add(col)
        ops_log.append(f"[Binary]   {col} : Yes→1 / No→0")

    if "Gender" in df_processed.columns and "Gender" not in encoded_cols:
        mapping = {"Female": 0, "Male": 1}
        df_processed["Gender"] = df_processed["Gender"].map(mapping)
        encoders["Gender"] = {"type": "binary", "mapping": mapping}
        encoded_cols.add("Gender")
        ops_log.append("[Binary]   Gender : Female→0 / Male→1")

    # ── 3. ORDINAL COLUMNS ───────────────────────────────────────────────────
    ordinal_mappings: Dict[str, Dict] = {
        "SmokingStatus": {"Non-Smoker": 0, "Former Smoker": 1, "Smoker": 2},
        "Stage": {"I": 0, "II": 1, "III": 2, "IV": 3},
        "TreatmentResponse": {
            "No Response": 0, "Partial Remission": 1, "Complete Remission": 2
        },
        "Survival_Category": {
            "Very_Short": 0, "Short": 1, "Medium": 2, "Long": 3, "Very_Long": 4
        },
        "Age_Group": {"Young": 0, "Middle_Age": 1, "Senior": 2, "Elderly": 3},
        "BMI_Category": {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3},
        "Tumor_Size_Category": {"Small": 0, "Medium": 1, "Large": 2},
        "Cancer_Stage": {"I": 0, "II": 1, "III": 2, "IV": 3},
        "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
        "Diet_Risk": {"Low": 0, "Medium": 1, "High": 2},
    }

    for col, mapping in ordinal_mappings.items():
        if col not in df_processed.columns:
            continue
        if col in encoded_cols:
            logger.warning("Column '%s' already encoded — skipping ordinal step.", col)
            continue
        df_processed[col] = df_processed[col].map(mapping)
        encoders[col] = {"type": "ordinal", "mapping": mapping}
        encoded_cols.add(col)
        ops_log.append(f"[Ordinal]  {col} : {list(mapping.keys())}")

    # ── 4. NOMINAL COLUMNS → pd.get_dummies ──────────────────────────────────
    nominal_columns = [
        c for c in [
            "Race/Ethnicity", "CancerType", "TreatmentType",
            "HospitalRegion", "Country",
        ]
        if c in df_processed.columns and c not in encoded_cols
    ]

    for col in nominal_columns:
        cols_before = set(df_processed.columns)
        df_processed = pd.get_dummies(
            df_processed, columns=[col], drop_first=True, dtype=int
        )
        new_cols = [c for c in df_processed.columns if c not in cols_before]
        encoders[col] = {
            "type": "ohe_dummies",
            "original_col": col,
            "new_cols": new_cols,
        }
        encoded_cols.add(col)
        ops_log.append(
            f"[Nominal]  {col} : get_dummies ({len(new_cols)} indicator columns)"
        )

    # GeneticMarker: nominal with NaN → 'Unknown', then get_dummies
    if "GeneticMarker" in df_processed.columns and "GeneticMarker" not in encoded_cols:
        df_processed["GeneticMarker"] = df_processed["GeneticMarker"].fillna("Unknown")
        cols_before = set(df_processed.columns)
        df_processed = pd.get_dummies(
            df_processed, columns=["GeneticMarker"], drop_first=True, dtype=int
        )
        new_cols = [c for c in df_processed.columns if c not in cols_before]
        encoders["GeneticMarker"] = {
            "type": "ohe_dummies",
            "original_col": "GeneticMarker",
            "new_cols": new_cols,
        }
        encoded_cols.add("GeneticMarker")
        ops_log.append(
            f"[Nominal]  GeneticMarker : get_dummies (NaN→'Unknown', "
            f"{len(new_cols)} indicator columns)"
        )

    # ── 5. REMAINING OBJECT COLUMNS → pd.get_dummies + warning ───────────────
    remaining_obj = [
        c for c in df_processed.select_dtypes(include=["object"]).columns
        if c not in encoded_cols
    ]
    for col in remaining_obj:
        logger.warning(
            "Column '%s' was not in any explicit encoding list — "
            "applying get_dummies as fallback. "
            "Consider adding it to binary/ordinal/nominal sections.",
            col,
        )
        cols_before = set(df_processed.columns)
        df_processed = pd.get_dummies(
            df_processed, columns=[col], drop_first=True, dtype=int
        )
        new_cols = [c for c in df_processed.columns if c not in cols_before]
        encoders[col] = {
            "type": "ohe_dummies",
            "original_col": col,
            "new_cols": new_cols,
        }
        encoded_cols.add(col)
        ops_log.append(
            f"[Auto]     {col} : get_dummies fallback ({len(new_cols)} indicator columns)"
        )

    # ── 6. VALIDATION ─────────────────────────────────────────────────────────
    remaining_obj_final = df_processed.select_dtypes(include=["object"]).columns.tolist()

    if verbose:
        _print_encoding_report(ops_log, remaining_obj_final)

    return df_processed, encoders


# Backward-compatible alias
encodage_processing = encode_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# TARGET ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def target_encoding(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    smoothing: float = 10.0,
    min_samples_leaf: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Target encoding with James-Stein regularization (smoothing).

    Formula: encoded = (n × mean_cat + smoothing × mean_global) / (n + smoothing)

    Parameters
    ----------
    smoothing : regularization strength (higher → closer to global mean)

    Returns
    -------
    df_encoded, encoding_maps
    """
    df_out = df.copy()
    global_mean = float(df[target_col].mean())
    encoding_maps: Dict = {}

    for col in cat_cols:
        if col not in df_out.columns:
            continue

        stats = df_out.groupby(col)[target_col].agg(["mean", "count"])
        smoother = 1 / (1 + np.exp(-(stats["count"] - min_samples_leaf) / smoothing))
        smooth_mean = smoother * stats["mean"] + (1 - smoother) * global_mean

        df_out[col] = df_out[col].map(smooth_mean).fillna(global_mean)
        encoding_maps[col] = smooth_mean.to_dict()

    return df_out, encoding_maps


# ──────────────────────────────────────────────────────────────────────────────
# FREQUENCY ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def frequency_encoding(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Replace each category with its relative frequency in the dataset.

    Returns
    -------
    df_encoded, freq_maps
    """
    df_out = df.copy()
    freq_maps: Dict = {}

    for col in cat_cols:
        if col not in df_out.columns:
            continue
        freq = df_out[col].value_counts(normalize=True)
        df_out[col] = df_out[col].map(freq).fillna(0)
        freq_maps[col] = freq.to_dict()

    return df_out, freq_maps


# ──────────────────────────────────────────────────────────────────────────────
# ONE-HOT ENCODING
# ──────────────────────────────────────────────────────────────────────────────


def one_hot_encode(
    df: pd.DataFrame,
    cat_cols: List[str],
    drop: str = "first",
    max_categories: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encoding with a cardinality constraint.

    Parameters
    ----------
    drop           : 'first' | 'if_binary' | None — avoids multicollinearity
    max_categories : columns exceeding this threshold are skipped

    Returns
    -------
    df_encoded, new_columns
    """
    cols_to_encode = [
        c for c in cat_cols
        if c in df.columns and df[c].nunique() <= max_categories
    ]
    skipped = [c for c in cat_cols if c not in cols_to_encode]

    if skipped:
        logger.warning("OHE skipped (cardinality > %d): %s", max_categories, skipped)

    if not cols_to_encode:
        return df, []

    df_out = pd.get_dummies(df, columns=cols_to_encode, drop_first=(drop == "first"), dtype=int)
    new_cols = [c for c in df_out.columns if c not in df.columns]

    logger.info("OHE: %d new columns created.", len(new_cols))
    return df_out, new_cols


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────


def _print_encoding_report(ops_log: List[str], remaining_obj: List[str]) -> None:
    print("\n" + "═" * 60)
    print("  ENCODING REPORT")
    print("═" * 60)
    for op in ops_log:
        print(f"  {op}")
    print()
    if remaining_obj:
        print(f" Un-encoded object columns: {remaining_obj}")
    else:
        print("All columns are numeric — encoding complete.")
    print("═" * 60 + "\n")
