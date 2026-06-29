"""
clinical_profiles.py — Patient risk profile generation.

Creates typical patient profiles for each risk level, enabling clinicians
to understand which data patterns correspond to which prediction level.

Features:
  - Model-importance-based profiles
  - Z-scores to contextualize feature values
  - Comparative profile visualization
  - Profile export to DataFrame

Public functions:
  - generate_patient_risk_profiles(model, feature_names, risk_levels,
                                    population_stats) → dict
  - compare_profiles_heatmap(risk_profiles, feature_names, top_n)       → None
  - profile_to_dataframe(risk_profiles, feature_names)                  → pd.DataFrame

Visualizations delegated to visualization.interpretability.profile_plots.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_framework.visualization.interpretability.profile_plots import (
    plot_risk_profiles,
    plot_profiles_heatmap,
)

logger = logging.getLogger("ml_framework.clinical_profiles")


# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_RISK_LEVELS: Dict[str, float] = {
    "Very Low":     0.10,
    "Low":          0.25,
    "Moderate":     0.55,
    "High":         0.75,
    "Very High":    0.90,
}


# =============================================================================
# PROFILE GENERATION
# =============================================================================


def generate_patient_risk_profiles(
    model,
    feature_names: List[str],
    risk_levels: Optional[Dict[str, float]] = None,
    population_stats: Optional[Dict[str, Dict[str, float]]] = None,
    plot: bool = True,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate typical patient profiles for each risk level.

    For each level, build a synthetic profile by adjusting the model's most
    important features in the direction of the target risk score.

    Parameters
    ----------
    model            : trained sklearn estimator
    feature_names    : ordered list of feature names
    risk_levels      : dict {label: target_risk_score}
                       (default: 5 levels from Very Low to Very High)
    population_stats : dict {
                           'mean': {feature: mean_value},
                           'std':  {feature: std_value}
                       } — reference population statistics
    plot             : display visualizations
    random_state     : seed for reproducibility

    Returns
    -------
    dict {risk_label: {'features': dict, 'risk_score': float, 'target_risk': float}}
    """
    np.random.seed(random_state)

    if risk_levels is None:
        risk_levels = _DEFAULT_RISK_LEVELS.copy()

    # Model importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).flatten()
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    importances = importances / (importances.sum() or 1.0)
    feature_importances = dict(zip(feature_names, importances))

    # Default population statistics
    if population_stats is None:
        population_stats = {
            "mean": {f: 0.0 for f in feature_names},
            "std":  {f: 1.0 for f in feature_names},
        }

    # Estimate feature risk directions
    feature_direction = _estimate_feature_directions(model, feature_names, population_stats)

    print("\n" + "═" * 60)
    print("  PATIENT RISK PROFILE GENERATION")
    print("═" * 60)

    risk_profiles: Dict[str, Dict[str, Any]] = {}
    sorted_features: List[Tuple[str, float]] = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )
    # Methodological considerations
    # The profiles generated below are SYNTHETIC APPROXIMATIONS for illustration purposes,
    # not validated patient archetypes. They start from the population mean and adjust
    # features via a heuristic factor (3.0 × std × importance × direction)
    # to shift the predicted score toward the target risk level.
    # 
    # Note on current limitations:
    #   1. The 3.0 scaling factor is a heuristic — it lacks clinical/statistical baseline.
    #   2. Adjusting features independently omits covariance; resulting profiles 
    #      may lack clinical plausibility (e.g., extreme outliers or conflicting metrics).
    # 
    # Recommended production alternative: select REAL patients from the target stratum 
    # (e.g., top 10% risk) and extract their percentile values as representative profiles.
    # 
    # WARNING: Use generate_patient_risk_profiles() for demonstration only, NOT for 
    # clinical decision-making. Profiles must always be validated by domain experts.

    logger.warning(
        "generate_patient_risk_profiles: producing synthetic approximations — "
        "not real patient archetypes. Use real patient percentile profiles "
        "for clinical applications."
    )
    print(
        "\n  ⚠  SYNTHETIC PROFILES — not real patients. "
        "Profiles are approximations for illustrative purposes only.\n"
        "     For clinical use, replace with real patient percentile profiles "
        "from the high-risk stratum.\n"
    )

    for level_name, target_risk in risk_levels.items():
        print(f"\n  Profile '{level_name}' (target: {target_risk:.0%})")

        # Start from population mean
        profile = {f: float(population_stats["mean"].get(f, 0.0)) for f in feature_names}

        n_to_modify = max(1, int(len(feature_names) * min(target_risk + 0.1, 0.6)))

        for feat, importance in sorted_features[:n_to_modify]:
            direction = feature_direction.get(feat, 1.0)
            std_dev   = population_stats["std"].get(feat, 1.0) or 1.0
            # factor 3.0 is an arbitrary heuristic — see caveat above
            adjustment = direction * (target_risk - 0.5) * 3.0 * std_dev * importance
            profile[feat] += adjustment

        # Compute actual risk score for this profile
        profile_arr = np.array([profile[f] for f in feature_names]).reshape(1, -1)
        profile_df  = pd.DataFrame(profile_arr, columns=feature_names)

        try:
            if hasattr(model, "predict_proba"):
                risk_score = float(model.predict_proba(profile_df)[0, 1])
            else:
                risk_score = float(model.predict(profile_df)[0])
        except Exception as exc:
            logger.warning("Risk score for '%s' could not be computed: %s", level_name, exc)
            risk_score = target_risk

        risk_profiles[level_name] = {
            "features":    profile,
            "risk_score":  risk_score,
            "target_risk": target_risk,
        }

        print(f"    Obtained score: {risk_score:.4f}")
        for feat, _ in sorted_features[:5]:
            val  = profile[feat]
            mean = population_stats["mean"].get(feat, 0.0)
            std  = population_stats["std"].get(feat, 1.0) or 1.0
            z    = (val - mean) / std
            print(f"      {feat[:30]:<30} : {val:.2f}  (z={z:+.2f})")

    if plot:
        plot_risk_profiles(risk_profiles, sorted_features, population_stats)

    return risk_profiles


# =============================================================================
# FEATURE DIRECTION ESTIMATION
# =============================================================================


def _estimate_feature_directions(
    model,
    feature_names: List[str],
    population_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Estimate whether increasing a feature raises (+1) or lowers (-1) the risk.
    Uses model coefficients when available, defaults to +1 otherwise.
    """
    directions: Dict[str, float] = {}

    if hasattr(model, "coef_"):
        coef    = model.coef_
        coef_arr = coef[0] if coef.ndim > 1 else coef
        for i, f in enumerate(feature_names):
            if i < len(coef_arr):
                directions[f] = float(np.sign(coef_arr[i])) or 1.0
            else:
                directions[f] = 1.0
    else:
        directions = {f: 1.0 for f in feature_names}

    return directions


# =============================================================================
# COMPARATIVE HEATMAP
# =============================================================================


def compare_profiles_heatmap(
    risk_profiles: Dict[str, Dict],
    feature_names: List[str],
    population_stats: Optional[Dict[str, Dict]] = None,
    top_n: int = 10,
) -> None:
    """
    Comparative z-score heatmap of top features across all risk profiles.

    Parameters
    ----------
    risk_profiles    : result of generate_patient_risk_profiles()
    feature_names    : list of features
    population_stats : population statistics (mean, std)
    top_n            : number of features to display
    """
    if population_stats is None:
        population_stats = {
            "mean": {f: 0.0 for f in feature_names},
            "std":  {f: 1.0 for f in feature_names},
        }

    sorted_levels = sorted(
        risk_profiles.keys(), key=lambda x: risk_profiles[x]["risk_score"]
    )
    top_features = feature_names[:top_n]

    matrix = []
    for level in sorted_levels:
        profile = risk_profiles[level]["features"]
        row = []
        for feat in top_features:
            val  = profile.get(feat, 0.0)
            mean = population_stats["mean"].get(feat, 0.0)
            std  = population_stats["std"].get(feat, 1.0) or 1.0
            row.append((val - mean) / std)
        matrix.append(row)

    df = pd.DataFrame(
        matrix,
        index=sorted_levels,
        columns=[f[:25] for f in top_features],
    )

    plot_profiles_heatmap(df, top_n)


# =============================================================================
# EXPORT
# =============================================================================


def profile_to_dataframe(
    risk_profiles: Dict[str, Dict],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Convert risk profiles to a tabular DataFrame.

    Parameters
    ----------
    risk_profiles : result of generate_patient_risk_profiles()
    feature_names : list of features

    Returns
    -------
    pd.DataFrame (profile × features + risk_score), indexed by profile name
    """
    rows = []
    for level, data in risk_profiles.items():
        row = {"profile": level, "risk_score": data["risk_score"]}
        for feat in feature_names:
            row[feat] = data["features"].get(feat, np.nan)
        rows.append(row)

    return pd.DataFrame(rows).set_index("profile")
