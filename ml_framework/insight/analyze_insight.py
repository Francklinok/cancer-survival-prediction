"""
analyze_insight.py — Automatic generation of data and model insights.

Provides synthetic interpretive summaries for:
  - Data (features, samples, quality)
  - Models (performance, best model)
  - Important features
  - MLOps recommendations
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_framework.analysis.eda import _cat_cols, _cramers_v, _num_cols
from ml_framework.visualization.analysis.importance_plots import plot_business_insights_bars
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.insight.analyze_insight")


# =============================================================================
# BUSINESS INSIGHTS
# =============================================================================


def business_insights(
    df: pd.DataFrame,
    target_col: str,
    domain: str = "medical",
    corr_threshold: float = 0.70,
    top_cat_features: int = 20,
    top_num_features: int = 20,
) -> pd.DataFrame:
    """
    Visual business insights from detected patterns.

    Insight types
    -------------
    risk_group        : subgroup with elevated/reduced target rate (visual bar)
    multicollinearity : numeric pairs with |r| > corr_threshold (visual only)
    discriminant      : numeric features that visually differ across target classes

    Note: Kruskal-Wallis with Bonferroni correction has been moved to
    statistical_analysis.compare_groups() for formal hypothesis testing.
    Here we only visualize the top discriminant features.

    Returns
    -------
    pd.DataFrame of detected visual patterns
    """
    section_header(f"BUSINESS INSIGHTS — {domain.upper()}")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")

    target_is_cat = (
        not pd.api.types.is_numeric_dtype(df[target_col])
        or df[target_col].nunique() <= 15
    )

    cat_cols_all = _cat_cols(df, exclude=target_col)
    num_cols_all = _num_cols(df, exclude=target_col)

    # Rank categorical columns by Cramér's V (visual selection only)
    cat_ranked: List[Tuple[str, float]] = []
    for col in cat_cols_all:
        try:
            v = _cramers_v(df[col].dropna(), df[target_col].dropna())
            cat_ranked.append((col, v))
        except Exception:
            pass
    cat_ranked.sort(key=lambda x: x[1], reverse=True)
    cat_cols = [c for c, _ in cat_ranked[:top_cat_features]]

    num_variance = df[num_cols_all].var().sort_values(ascending=False)
    num_cols     = num_variance.head(top_num_features).index.tolist()

    def _log_ratio(p_max: float, p_min: float, eps: float = 1e-3) -> float:
        return round(float(np.log((p_max + eps) / (p_min + eps))), 4)

    insights_rows: List[Dict] = []

    # Insight 1: Risk groups (visual subgroup rates)
    if target_is_cat and cat_cols:
        for col in cat_cols:
            try:
                group_rates = (
                    df.groupby(col)[target_col]
                    .value_counts(normalize=True).unstack(fill_value=0.0)
                )
                for tv in df[target_col].dropna().unique():
                    if tv not in group_rates.columns:
                        continue
                    col_vals  = group_rates[tv]
                    if len(col_vals) < 2:
                        continue
                    max_group = col_vals.idxmax()
                    min_group = col_vals.idxmin()
                    log_r     = _log_ratio(col_vals.max(), col_vals.min())
                    if log_r > 0.40:
                        insights_rows.append({
                            "type":    "risk_group",
                            "feature": col,
                            "detail":  (
                                f"{col}='{max_group}' → {target_col}='{tv}' "
                                f"{col_vals.max()*100:.1f}% vs '{min_group}' {col_vals.min()*100:.1f}%"
                            ),
                            "score":   log_r,
                            "interpretation": f"Subgroup effect log-ratio={log_r:.2f}",
                        })
            except Exception as exc:
                logger.debug("Risk group failed for %s: %s", col, exc)

    # Insight 2: Multicollinearity (visual)
    if len(num_cols) >= 2:
        try:
            corr  = df[num_cols].corr(method="pearson").abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            strong = upper.stack()
            strong = strong[strong > corr_threshold].sort_values(ascending=False)
            for (c1, c2), val in strong.items():
                insights_rows.append({
                    "type":    "multicollinearity",
                    "feature": f"{c1} × {c2}",
                    "detail":  f"|r| = {val:.3f}",
                    "score":   round(float(val), 4),
                    "interpretation": f"|r|={val:.2f} > {corr_threshold} — multicollinearity risk",
                })
        except Exception as exc:
            logger.warning("Correlation computation failed: %s", exc)

    # Insight 3: Discriminant features (visual — group mean spread only)
    if target_is_cat and num_cols:
        for col in num_cols:
            try:
                means    = df.groupby(target_col)[col].mean()
                spread   = float(means.max() - means.min())
                std_pool = float(df[col].std())
                if std_pool > 0 and spread / std_pool > 0.3:
                    insights_rows.append({
                        "type":    "discriminant",
                        "feature": col,
                        "detail":  f"group mean spread={spread:.3f} ({spread/std_pool:.2f}σ)",
                        "score":   round(spread / std_pool, 4),
                        "interpretation": f"'{col}' shows visual group separation ({spread/std_pool:.2f}σ)",
                    })
            except Exception:
                pass

    if not insights_rows:
        print("\n No significant visual patterns detected.")
        return pd.DataFrame(columns=["type", "feature", "detail", "score", "interpretation"])

    ins_df = (
        pd.DataFrame(insights_rows)
        .sort_values(["type", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    type_icons = {"risk_group": "📌", "multicollinearity": "🔗", "discriminant": "📊"}
    for ins_type in ins_df["type"].unique():
        subset = ins_df[ins_df["type"] == ins_type]
        icon   = type_icons.get(ins_type, "•")
        print(f"\n  {icon} {ins_type.upper()} ({len(subset)} pattern(s)):")
        for _, row in subset.head(10).iterrows():
            print(f"    • {row['detail']}")
            print(f"      → {row['interpretation']}")

    if target_is_cat and cat_ranked:
        plot_business_insights_bars(df, cat_ranked[:3], target_col)

    print("\n  → For formal statistical tests: statistical_analysis.compare_groups()")
    return ins_df


# =============================================================================
# DATA INSIGHTS
# =============================================================================


def analyze_data_insights(
    data_original: pd.DataFrame,
    data_engineered: Optional[pd.DataFrame] = None,
    data_final: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
) -> Dict:
    """
    Qualitative summary of data transformation steps.

    Parameters
    ----------
    data_original   : raw data after loading
    data_engineered : data after feature engineering (optional)
    data_final      : final data before modeling (optional)
    target_col      : name of the target column

    Returns
    -------
    dict with metrics and interpretations
    """
    n_orig_feat = data_original.shape[1]
    n_orig_rows = data_original.shape[0]

    insights = {
        "original_features": n_orig_feat,
        "original_samples":  n_orig_rows,
        "engineered_features": data_engineered.shape[1] if data_engineered is not None else None,
        "final_features": (
            (data_final.shape[1] - (1 if target_col else 0))
            if data_final is not None else None
        ),
        "final_samples": data_final.shape[0] if data_final is not None else None,
    }

    missing_rate = data_original.isnull().mean().mean()
    insights["missing_rate_original"] = round(float(missing_rate * 100), 2)
    insights["data_quality_score"]    = round(float(max(0, 1 - missing_rate) * 100), 1)

    if data_engineered is not None:
        delta = data_engineered.shape[1] - n_orig_feat
        insights["features_added_by_engineering"] = delta
        insights["feature_enrichment_pct"] = round(delta / n_orig_feat * 100, 1)

    if data_final is not None and data_engineered is not None:
        n_final = data_final.shape[1] - (1 if target_col else 0)
        n_eng   = data_engineered.shape[1]
        insights["features_removed_by_selection"] = max(0, n_eng - n_final)

    return insights


# =============================================================================
# MODEL INSIGHTS
# =============================================================================


def analyze_model_insights(
    evaluation_results: Dict,
    best_model_name: Optional[str] = None,
) -> Dict:
    """
    Summary of performance for all trained models.

    Parameters
    ----------
    evaluation_results : dict {model_name: {cv_mean, cv_std, test_score, ...}}
    best_model_name    : name of the best model (optional)

    Returns
    -------
    dict with: models_tested, best_model, performance_range, interpretations
    """
    if not evaluation_results:
        return {}

    scores = {
        name: res.get("cv_mean", res.get("test_score", 0))
        for name, res in evaluation_results.items()
    }
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    insights = {
        "models_tested": len(evaluation_results),
        "best_model":    best_model_name or sorted_models[0][0],
        "best_score":    round(sorted_models[0][1], 4),
        "worst_score":   round(sorted_models[-1][1], 4),
        "performance_range": {
            "min":    round(min(scores.values()), 4),
            "max":    round(max(scores.values()), 4),
            "spread": round(max(scores.values()) - min(scores.values()), 4),
        },
        "ranking": [(name, round(sc, 4)) for name, sc in sorted_models],
    }

    best_sc = sorted_models[0][1]
    insights["performance_interpretation"] = (
        "Excellent — production-level performance (AUC >= 0.95)"   if best_sc >= 0.95
        else "Very good — high-performing model (AUC >= 0.90)"     if best_sc >= 0.90
        else "Good — acceptable, improvements possible (AUC >= 0.80)" if best_sc >= 0.80
        else "Moderate — optimization required (AUC < 0.80)"
    )

    return insights


# =============================================================================
# FEATURE INSIGHTS
# =============================================================================


def analyze_feature_insights(
    importance_df: pd.DataFrame,
    top_n: int = 10,
) -> Dict:
    """
    Analyze feature importance to extract predictive insight.

    Parameters
    ----------
    importance_df : DataFrame with columns 'Feature' and 'Importance'
    top_n         : number of top features

    Returns
    -------
    dict
    """
    if importance_df is None or importance_df.empty:
        return {}

    top       = importance_df.head(top_n)
    total_imp = importance_df["Importance"].sum()
    top_concentration = top["Importance"].sum() / total_imp if total_imp > 0 else 0

    return {
        "top_features":        top["Feature"].tolist(),
        "top_n_concentration": round(float(top_concentration * 100), 2),
        "most_important":      top["Feature"].iloc[0],
        "n_total_features":    len(importance_df),
        "interpretation": (
            f"The top {top_n} most important features concentrate "
            f"{top_concentration * 100:.1f}% of the predictive information. "
            f"The dominant feature is '{top['Feature'].iloc[0]}'."
        ),
    }


# =============================================================================
# PIPELINE REPORT
# =============================================================================


def generate_pipeline_report(
    data_insights: Dict,
    model_insights: Dict,
    feature_insights: Optional[Dict] = None,
    verbose: bool = True,
) -> str:
    """
    Generate a synthetic text report of the entire pipeline.

    Returns
    -------
    str — formatted report
    """
    lines = [
        "",
        "╔" + "═" * 62 + "╗",
        "║{:^62}║".format("  SYNTHETIC REPORT OF THE MEDICAL ML PIPELINE  "),
        "╠" + "═" * 62 + "╣",
        "",
        "  ─── DATA ────────────────────────────────────────────────",
        f"  Original features        : {data_insights.get('original_features', '?')}",
        f"  Samples                  : {data_insights.get('original_samples', '?')}",
        f"  Initial NaN rate         : {data_insights.get('missing_rate_original', '?')} %",
        f"  Data quality score       : {data_insights.get('data_quality_score', '?')} / 100",
    ]

    if data_insights.get("features_added_by_engineering"):
        lines.append(
            f"  Features created (eng.)  : +{data_insights['features_added_by_engineering']}"
            f" ({data_insights.get('feature_enrichment_pct', '?')} %)"
        )

    if data_insights.get("final_features"):
        lines.append(f"  Final features           : {data_insights['final_features']}")

    if model_insights:
        lines += [
            "",
            "  ─── MODELS ──────────────────────────────────────────────",
            f"  Models tested            : {model_insights.get('models_tested', '?')}",
            f"  Best model               : {model_insights.get('best_model', '?')}",
            f"  Best CV score            : {model_insights.get('best_score', '?')}",
            f"  Interpretation           : {model_insights.get('performance_interpretation', '?')}",
        ]
        if model_insights.get("ranking"):
            lines.append("  Ranking:")
            for rank, (name, sc) in enumerate(model_insights["ranking"], 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                lines.append(f"    {medal} {rank}. {name:<35} {sc:.4f}")

    if feature_insights:
        lines += [
            "",
            "  ─── FEATURES ───────────────────────────────────────────",
            f"  {feature_insights.get('interpretation', '')}",
        ]

    lines += ["", "╚" + "═" * 62 + "╝", ""]
    report = "\n".join(lines)

    if verbose:
        print(report)

    return report
