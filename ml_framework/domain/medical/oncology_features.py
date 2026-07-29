"""
oncology_features.py — Clinical interaction features for the oncology
reference dataset, grounded in domain knowledge (not generic arithmetic).

Moved here verbatim from what used to be
ml_framework.features.feature_engineering._add_medical_features(), applied
unconditionally to every dataset regardless of domain. Same formulas, same
column names, same behaviour — now an explicit opt-in via
add_oncology_features() (or engineer_features(domain_features_fn=...))
instead of a framework-wide default.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

_STAGE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}
_SMOKING_MAP = {"Non-Smoker": 0, "Former Smoker": 1, "Smoker": 2}
_FAMILY_MAP = {"No": 0, "Yes": 1}

_SURVIVAL_LEAKAGE_COLS = {"SurvivalMonths", "Survival_Months", "survival_months"}


def add_oncology_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> List[str]:
    """
    Clinical interaction features grounded in oncology domain knowledge.
    """
    new_cols: List[str] = []

    def _asnum(series: pd.Series, mapping: dict) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(0).astype(float)
        return series.map(mapping).fillna(0).astype(float)

    # 1. Age × Stage — clinical aggressiveness proxy
    if "Age" in df.columns and "Stage" in df.columns:
        df["age_stage_risk"] = df["Age"].fillna(0) * _asnum(df["Stage"], _STAGE_MAP)
        new_cols.append("age_stage_risk")

    # 2. BMI × SmokingStatus — comorbidity burden
    if "BMI" in df.columns and "SmokingStatus" in df.columns:
        df["bmi_smoking_risk"] = df["BMI"].fillna(0) * _asnum(df["SmokingStatus"], _SMOKING_MAP)
        new_cols.append("bmi_smoking_risk")

    # 3. TumorSize × Stage — tumour aggressiveness
    tumor_col = next(
        (c for c in ["TumorSize", "Tumor_Size_cm", "Tumor_Size"] if c in df.columns), None
    )
    if tumor_col and "Stage" in df.columns:
        df["tumor_aggressiveness"] = (
            df[tumor_col].fillna(0) * _asnum(df["Stage"], _STAGE_MAP)
        )
        new_cols.append("tumor_aggressiveness")

    # 4. Composite risk score (normalized components, 0-1 each)
    risk_parts: List[pd.Series] = []
    if "Stage" in df.columns:
        risk_parts.append(_asnum(df["Stage"], _STAGE_MAP) / 4.0)
    if "SmokingStatus" in df.columns:
        risk_parts.append(_asnum(df["SmokingStatus"], _SMOKING_MAP) / 2.0)
    if "FamilyHistory" in df.columns:
        risk_parts.append(_asnum(df["FamilyHistory"], _FAMILY_MAP))
    if "Age" in df.columns:
        age_s = df["Age"].fillna(0).astype(float)
        mx = age_s.max()
        risk_parts.append(age_s / mx if mx > 0 else age_s)

    if len(risk_parts) >= 2:
        df["cumulative_risk_score"] = sum(risk_parts)
        new_cols.append("cumulative_risk_score")

    # 5. SurvivalMonths normalised — skipped when it's a leakage risk
    surv_col = next(
        (c for c in ["SurvivalMonths", "Survival_Months"] if c in df.columns), None
    )
    if surv_col and surv_col not in _SURVIVAL_LEAKAGE_COLS:
        mx = df[surv_col].max()
        if mx > 0:
            df["survival_rate_normalized"] = df[surv_col] / mx
            new_cols.append("survival_rate_normalized")

    return new_cols
