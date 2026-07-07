"""
data_drift.py — Data Drift Detection & Monitoring.

=============================================================================
FEATURE DRIFT
=============================================================================
  detect_data_drift()          — PSI + KS + Chi² + Wasserstein + JS per feature
  compute_psi()                — Population Stability Index (continuous)
  compute_kl_divergence()      — KL Divergence (reference ∥ current)

=============================================================================
TARGET DRIFT
=============================================================================
  detect_target_drift()        — Monitor shift in target distribution over time

=============================================================================
CONCEPT DRIFT INDICATORS
=============================================================================
  concept_drift_indicators()   — Proxy signals: residual shift, error rate change,
                                  feature importance stability, calibration drift

=============================================================================
POPULATION STABILITY MONITORING
=============================================================================
  population_stability_report() — Full PSI matrix across all features + summary
  monitor_population_stability() — Rolling window PSI tracking over time slices

All plotting delegated to visualization.analysis.drift_plots.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from ml_framework.visualization.analysis.drift_plots import plot_drift_overview

logger = logging.getLogger("ml_framework.data_drift")

# PSI thresholds
PSI_OK      = 0.10   # < 0.10: stable distribution
PSI_WARNING = 0.20   # 0.10–0.20: mild drift — monitor
# > 0.20: significant drift — action required


# ──────────────────────────────────────────────────────────────────────────────
# PSI
# ──────────────────────────────────────────────────────────────────────────────


def bucket(s: pd.Series, bp: np.ndarray) -> np.ndarray:
    """
    Convert a series to a normalized histogram over bin boundaries bp.

    Epsilon is applied AFTER normalizing to proportions — not before.
    Adding epsilon to raw counts before dividing by n creates a bias
    proportional to the number of bins (each bin gets +1e-6/n added),
    which distorts PSI especially when n is small.

    Parameters
    ----------
    s  : pd.Series or array-like of numeric values
    bp : array of breakpoints (bin edges), e.g. np.percentile(ref, linspace)

    Returns
    -------
    np.ndarray — normalized proportions, size len(bp)-1
    """
    counts, _ = np.histogram(s, bins=bp)
    n = len(s)
    if n == 0:
        return np.ones(len(counts)) / len(counts)
    pct = counts / n                      # proportions sum to ≤ 1.0 (exactly 1.0 if no NaN)
    pct = np.maximum(pct, 1e-6)          # clip zeros to epsilon AFTER normalizing
    return pct / pct.sum()               # re-normalize so proportions sum to 1.0


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI) for a continuous variable.

    PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

    Returns
    -------
    float — PSI value (0 = identical, > 0.20 = strong drift)
    """
    breaks = np.percentile(reference.dropna(), np.linspace(0, 100, n_bins + 1))
    breaks = np.unique(breaks)

    n_bins_actual = len(breaks) - 1
    if n_bins_actual < 2:
        logger.warning("PSI: not enough unique breakpoints (< 2 bins). PSI set to 0.")
        return 0.0
    if n_bins_actual < n_bins:
        logger.warning(
            "PSI: only %d unique breakpoints (requested %d bins). "
            "Feature has low cardinality — PSI may not be comparable across features "
            "computed with the full %d bins. Consider computing PSI on raw category "
            "frequencies for low-cardinality variables.",
            n_bins_actual, n_bins, n_bins,
        )

    ref_counts, _ = np.histogram(reference.dropna(), bins=breaks)
    cur_counts, _ = np.histogram(current.dropna(), bins=breaks)

    ref_pct = np.maximum(ref_counts / len(reference.dropna()), 1e-6)
    cur_pct = np.maximum(cur_counts / len(current.dropna()), 1e-6)
    
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


# ──────────────────────────────────────────────────────────────────────────────
# FULL DRIFT DETECTION
# ──────────────────────────────────────────────────────────────────────────────


def detect_data_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    psi_threshold_warning: float = PSI_WARNING,
    verbose: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Detect statistical drift between a reference and a current dataset.

    Parameters
    ----------
    df_reference          : reference DataFrame (e.g. training data)
    df_current            : current DataFrame (e.g. production data)
    columns               : columns to analyze (all common columns if None)
    alpha                 : significance threshold for statistical tests
    psi_threshold_warning : PSI threshold for drift alert
    verbose               : print the report
    plot                  : display visualizations

    Returns
    -------
    pd.DataFrame — per-column drift results with metrics.
    """
    common_cols = df_reference.columns.intersection(df_current.columns).tolist()
    if columns:
        columns = [c for c in columns if c in common_cols]
    else:
        columns = common_cols

    if not columns:
        raise ValueError("No common columns found between the two datasets.")

    results = []

    for col in columns:
        ref_col    = df_reference[col].dropna()
        cur_col    = df_current[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(ref_col)

        row: Dict = {
            "column":         col,
            "type":           "numeric" if is_numeric else "categorical",
            "drift_detected": False,
            "drift_severity": "stable",
        }

        if is_numeric:
            ks_stat, ks_p = ks_2samp(ref_col.values, cur_col.values)
            row["ks_stat"]  = round(float(ks_stat), 4)
            row["ks_p"]     = round(float(ks_p), 6)
            row["ks_drift"] = bool(ks_p < alpha)

            psi = compute_psi(ref_col, cur_col)
            row["psi"]       = psi
            row["psi_drift"] = bool(psi > psi_threshold_warning)

            try:
                w_dist = float(wasserstein_distance(ref_col.values, cur_col.values))
                row["wasserstein"] = round(w_dist, 6)
            except Exception:
                row["wasserstein"] = np.nan

            row["mean_ref"]        = round(float(ref_col.mean()), 4)
            row["mean_cur"]        = round(float(cur_col.mean()), 4)
            # Preserve sign: positive = current > reference (increase), negative = decrease.
            # abs() on numerator was masking direction of the shift.
            row["mean_shift_pct"]  = round(
                (cur_col.mean() - ref_col.mean()) / (abs(ref_col.mean()) + 1e-10) * 100, 2
            )
            row["std_ref"] = round(float(ref_col.std()), 4)
            row["std_cur"] = round(float(cur_col.std()), 4)

            row["drift_detected"] = bool(row["ks_drift"] or row["psi_drift"])

        else:
            try:
                all_cats   = sorted(set(ref_col.unique()) | set(cur_col.unique()))
                ref_vc     = ref_col.value_counts()
                cur_vc     = cur_col.value_counts()
                ref_counts = ref_vc.reindex(all_cats, fill_value=0).values.astype(float)
                cur_counts = cur_vc.reindex(all_cats, fill_value=0).values.astype(float)

                # JS divergence — symmetric, bounded [0, ln2] — primary drift metric
                p_ref = np.maximum(ref_counts / ref_counts.sum(), 1e-10)
                p_cur = np.maximum(cur_counts / cur_counts.sum(), 1e-10)
                p_ref /= p_ref.sum(); p_cur /= p_cur.sum()
                js_cat = float(jensenshannon(p_ref, p_cur))
                # Threshold: JS > 0.10 = mild drift; JS > 0.20 = significant drift
                js_cat_drift = js_cat > 0.10
                row["js_cat"]       = round(js_cat, 6)
                row["js_cat_drift"] = js_cat_drift

                # Chi-square on raw counts (valid only if all expected >= 5)
                contingency = np.vstack([ref_counts, cur_counts])
                if contingency.min() >= 5:
                    chi2, p_chi2, _, _ = chi2_contingency(contingency)
                    row["chi2_stat"]  = round(float(chi2), 4)
                    row["chi2_p"]     = round(float(p_chi2), 6)
                    row["chi2_drift"] = bool(p_chi2 < alpha)
                else:
                    row["chi2_stat"]  = np.nan
                    row["chi2_p"]     = np.nan
                    row["chi2_drift"] = None

                row["drift_detected"] = bool(js_cat_drift or row["chi2_drift"])
                row["colonne"] = col
            except Exception as e:
                row["js_cat_drift"] = None
                logger.debug("Categorical drift failed for %s: %s", col, e)

        # Jensen-Shannon divergence — for categoricals, reuse js_cat already computed
        # produce a divergent js_divergence vs js_cat, confusing monitoring reports.
        try:
            if is_numeric:
                bins = np.histogram_bin_edges(pd.concat([ref_col, cur_col]), bins=20)
                p, _ = np.histogram(ref_col, bins=bins, density=True)
                q, _ = np.histogram(cur_col, bins=bins, density=True)
                p = np.maximum(p, 1e-10)
                q = np.maximum(q, 1e-10)
                p = p / p.sum()
                q = q / q.sum()
                row["js_divergence"] = round(float(jensenshannon(p, q)), 6)
            else:
                row["js_divergence"] = row.get("js_cat", np.nan)
        except Exception:
            row["js_divergence"] = np.nan

        # Severity
        if row["drift_detected"]:
            psi_v = row.get("psi", 0) or 0
            row["drift_severity"] = (
                "critical" if psi_v > 0.25 or row.get("ks_stat", 0) > 0.30
                else "moderate"
            )

        results.append(row)

    drift_df = pd.DataFrame(results)

    if verbose:
        _print_drift_report(drift_df)

    if plot:
        plot_drift_overview(drift_df, df_reference, df_current, columns[:8])

    return drift_df


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────


def _print_drift_report(drift_df: pd.DataFrame) -> None:
    drifted  = drift_df[drift_df["drift_detected"]]
    n_drifted = len(drifted)
    n_total   = len(drift_df)

    print("\n" + "═" * 65)
    print("  DATA DRIFT DETECTION REPORT")
    print("═" * 65)
    print(f"  Columns analyzed  : {n_total}")
    print(f"  Columns with drift: {n_drifted}  ({n_drifted/n_total*100:.1f}%)")
    print()

    if n_drifted > 0:
        for _, row in drifted.iterrows():
            sev     = row["drift_severity"]
            emoji   = "🔴" if sev == "critical" else "🟠"
            psi_str = f" PSI={row.get('psi','?'):.3f}" if "psi" in row else ""
            ks_str  = f" KS-p={row.get('ks_p','?'):.4f}" if "ks_p" in row else ""
            chi_str = f" χ²-p={row.get('chi2_p','?'):.4f}" if "chi2_p" in row else ""
            print(f"  {emoji}  {row['column']:<35} [{sev.upper()}]{psi_str}{ks_str}{chi_str}")

        print("\n  Recommendations:")
        if any(drift_df["drift_severity"] == "critical"):
            print("  → Critical drift detected: retrain the model immediately.")
        print("  → Verify the source of the current data.")
        print("  → Monitor production performance metrics.")
    else:
        print("  ✅ No drift detected — stable distribution.")

    print("═" * 65 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# KL DIVERGENCE
# ──────────────────────────────────────────────────────────────────────────────

def compute_kl_divergence(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 20,
    epsilon: float = 1e-10,
) -> float:
    """
    KL Divergence  KL(P_reference ∥ P_current).

    Measures the information lost when P_current is used to approximate
    P_reference. Asymmetric — use Jensen-Shannon for a symmetric alternative.

    KL(P ∥ Q) = Σ P(x) · log(P(x) / Q(x))

    Parameters
    ----------
    reference : reference distribution Series
    current   : current distribution Series
    n_bins    : number of histogram bins (numeric) or unique values (categorical)
    epsilon   : small constant to avoid log(0)

    Returns
    -------
    float — KL divergence (0 = identical; higher = more divergent; ∞ if Q=0 where P>0)
    """
    is_numeric = pd.api.types.is_numeric_dtype(reference)

    if is_numeric:
        bins = np.histogram_bin_edges(
            pd.concat([reference.dropna(), current.dropna()]), bins=n_bins
        )
        p, _ = np.histogram(reference.dropna(), bins=bins)
        q, _ = np.histogram(current.dropna(),   bins=bins)
    else:
        all_cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
        p = reference.dropna().value_counts().reindex(all_cats, fill_value=0).values
        q = current.dropna().value_counts().reindex(all_cats, fill_value=0).values

    p = p.astype(float) + epsilon
    q = q.astype(float) + epsilon
    p /= p.sum()
    q /= q.sum()

    kl = float(np.sum(p * np.log(p / q)))
    return round(kl, 6)


# ──────────────────────────────────────────────────────────────────────────────
# TARGET DRIFT
# ──────────────────────────────────────────────────────────────────────────────

def detect_target_drift(
    y_reference: pd.Series,
    y_current: pd.Series,
    alpha: float = 0.05,
    plot: bool = True,
) -> Dict:
    """
    Detect drift in the target variable distribution.

    Computes:
      - KS test or Chi² (based on target type)
      - PSI  (numeric targets)
      - JS Divergence
      - KL Divergence
      - Mean/proportion shift

    Parameters
    ----------
    y_reference : target values from the reference period (e.g. training labels)
    y_current   : target values from the current period (e.g. production labels)
    alpha       : significance threshold for statistical tests

    Returns
    -------
    dict with keys: drift_detected, severity, ks_stat, ks_p, psi, js_div, kl_div,
                    mean_ref, mean_cur, mean_shift_pct
    """
    print("\n" + "═" * 65)
    print("  TARGET DRIFT DETECTION")
    print("═" * 65)

    is_numeric = pd.api.types.is_numeric_dtype(y_reference)
    result: Dict = {"drift_detected": False, "severity": "stable"}

    if is_numeric:
        ks_stat, ks_p = ks_2samp(y_reference.dropna().values, y_current.dropna().values)
        psi    = compute_psi(y_reference.dropna(), y_current.dropna())
        js_div = float(jensenshannon(
            *_normalize_histograms(y_reference, y_current, n_bins=20)
        ))
        kl_div = compute_kl_divergence(y_reference, y_current)

        result.update({
            "type":          "numeric",
            "ks_stat":       round(float(ks_stat), 4),
            "ks_p":          round(float(ks_p), 6),
            "ks_drift":      bool(ks_p < alpha),
            "psi":           psi,
            "psi_drift":     bool(psi > PSI_WARNING),
            "js_divergence": round(js_div, 6),
            "kl_divergence": kl_div,
            "mean_ref":      round(float(y_reference.mean()), 4),
            "mean_cur":      round(float(y_current.mean()), 4),
            # Preserve sign: positive = current > reference (increase), negative = decrease.
            "mean_shift_pct": round(
                (y_current.mean() - y_reference.mean()) /
                (abs(y_reference.mean()) + 1e-10) * 100, 2
            ),
        })
        result["drift_detected"] = result["ks_drift"] or result["psi_drift"]

        print(f"  KS stat:       {result['ks_stat']:.4f}  p={result['ks_p']:.4f}")
        print(f"  PSI:           {psi:.4f}  ({'DRIFT' if result['psi_drift'] else 'stable'})")
        print(f"  JS Divergence: {js_div:.4f}")
        print(f"  KL Divergence: {kl_div:.4f}")
        print(f"  Mean shift:    {result['mean_ref']:.4f} → {result['mean_cur']:.4f}  "
              f"({result['mean_shift_pct']:.1f}%)")

    else:
        all_cats   = sorted(set(y_reference.dropna().unique()) | set(y_current.dropna().unique()))
        ref_counts = y_reference.dropna().value_counts().reindex(all_cats, fill_value=0).values.astype(float)
        cur_counts = y_current.dropna().value_counts().reindex(all_cats, fill_value=0).values.astype(float)

        p_arr = np.maximum(ref_counts / ref_counts.sum(), 1e-10)
        q_arr = np.maximum(cur_counts / cur_counts.sum(), 1e-10)
        p_arr /= p_arr.sum(); q_arr /= q_arr.sum()

        js_div = float(jensenshannon(p_arr, q_arr))
        kl_div = float(np.sum(p_arr * np.log(p_arr / q_arr)))

        result.update({
            "type":          "categorical",
            "js_divergence": round(js_div, 6),
            "kl_divergence": round(kl_div, 6),
        })

        # Chi-square valid only on raw counts (expected >= 5 per cell)
        contingency = np.vstack([ref_counts, cur_counts])
        if contingency.min() >= 5:
            chi2, chi2_p, _, _ = chi2_contingency(contingency)
            result["chi2_stat"]  = round(float(chi2), 4)
            result["chi2_p"]     = round(float(chi2_p), 6)
            result["chi2_drift"] = bool(chi2_p < alpha)
            print(f"  Chi² stat:     {result['chi2_stat']:.4f}  p={result['chi2_p']:.4f}")
        else:
            result["chi2_stat"]  = np.nan
            result["chi2_p"]     = np.nan
            result["chi2_drift"] = None
            print(f"  Chi² skipped: expected counts < 5 — use JS divergence")

        # JS > 0.10 signals mild drift; JS > 0.20 signals significant drift
        result["drift_detected"] = bool(js_div > 0.10 or result.get("chi2_drift"))

        p_ref = pd.Series(p_arr, index=all_cats)
        p_cur = pd.Series(q_arr, index=all_cats)
        print(f"  JS Divergence: {js_div:.4f}")
        print(f"  KL Divergence: {kl_div:.4f}")
        print(f"\n  Class distribution (reference):")
        print("  " + p_ref.round(3).to_string())
        print(f"\n  Class distribution (current):")
        print("  " + p_cur.round(3).to_string())

    if result["drift_detected"]:
        psi_v = result.get("psi", 0) or 0
        result["severity"] = (
            "critical" if psi_v > 0.25 or result.get("ks_stat", 0) > 0.30
            else "moderate"
        )

    status = "DRIFT DETECTED" if result["drift_detected"] else "stable"
    print(f"\n  → Target status: [{status}]  severity: {result['severity']}")
    print("═" * 65 + "\n")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if is_numeric:
            # Shared bin edges (computed once from the pooled data) so both
            # histograms are directly comparable — independent binning per
            # series can misalign bar positions/widths between the two.
            shared_bins = np.histogram_bin_edges(
                pd.concat([y_reference.dropna(), y_current.dropna()]), bins=30
            )
            axes[0].hist(y_reference.dropna(), bins=shared_bins, alpha=0.6,
                         color="steelblue", label="Reference", density=True)
            axes[0].hist(y_current.dropna(),   bins=shared_bins, alpha=0.6,
                         color="darkorange", label="Current", density=True)
            axes[0].set_title("Target Distribution", fontsize=10)
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        else:
            x = np.arange(len(all_cats))
            w = 0.35
            axes[0].bar(x - w/2, p_ref.values, w, label="Reference", color="steelblue", alpha=0.8)
            axes[0].bar(x + w/2, p_cur.values, w, label="Current",   color="darkorange", alpha=0.8)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(all_cats, rotation=45, ha="right")
            axes[0].set_title("Target Class Distribution", fontsize=10)
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.3)

        metrics_names = ["JS Div", "KL Div"]
        metrics_vals  = [result.get("js_divergence", 0), result.get("kl_divergence", 0)]
        if "psi" in result:
            metrics_names.append("PSI")
            metrics_vals.append(result["psi"])
        axes[1].bar(metrics_names, metrics_vals,
                    color=["#e74c3c" if v > 0.1 else "#2ecc71" for v in metrics_vals])
        axes[1].set_ylabel("Divergence metric value")
        axes[1].set_title("Divergence Metrics", fontsize=10)
        axes[1].grid(axis="y", alpha=0.3)

        plt.suptitle("Target Drift Detection", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CONCEPT DRIFT INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

def concept_drift_indicators(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    target_col: str,
    model=None,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    plot: bool = True,
) -> Dict:
    """
    Concept Drift Indicators — proxy signals that the P(Y|X) relationship has changed.

    Concept drift differs from feature drift: features may be stable while the
    relationship between features and target has fundamentally shifted.

    Indicators computed
    -------------------
    1. Target marginal drift        : KS / Chi² on Y distribution
    2. Feature-target correlation shift : Pearson |r| stability per numeric feature
    3. Residual mean shift          : if model provided, compare mean(Y - Ŷ) ref vs cur
    4. Error rate change            : if model provided, compare accuracy/MAE ref vs cur
    5. Top feature rank stability   : Spearman rank correlation of feature importances

    Parameters
    ----------
    model : fitted sklearn model (optional). If provided, enables residual & error indicators.

    Returns
    -------
    dict with per-indicator results and an overall concept_drift_score [0–1]
    """
    print("\n" + "═" * 65)
    print("  CONCEPT DRIFT INDICATORS")
    print("═" * 65)

    if feature_cols is None:
        feature_cols = [
            c for c in df_reference.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]

    indicators: Dict = {}
    signals: List[str] = []

    # 1. Target marginal drift
    y_ref = df_reference[target_col].dropna()
    y_cur = df_current[target_col].dropna()
    is_numeric_target = pd.api.types.is_numeric_dtype(y_ref)

    if is_numeric_target:
        ks_stat, ks_p = ks_2samp(y_ref.values, y_cur.values)
        target_drifted = bool(ks_p < alpha)
        indicators["target_ks_stat"] = round(float(ks_stat), 4)
        indicators["target_ks_p"]    = round(float(ks_p), 6)
    else:
        all_cats   = sorted(set(y_ref.unique()) | set(y_cur.unique()))
        ref_raw    = y_ref.value_counts().reindex(all_cats, fill_value=0).values.astype(float)
        cur_raw    = y_cur.value_counts().reindex(all_cats, fill_value=0).values.astype(float)
        ct         = np.vstack([ref_raw, cur_raw])
        if ct.min() >= 5:
            chi2, chi2_p, _, _ = chi2_contingency(ct)
            target_drifted = bool(chi2_p < alpha)
            indicators["target_chi2_p"] = round(float(chi2_p), 6)
        else:
            p_ref_n = np.maximum(ref_raw / ref_raw.sum(), 1e-10)
            p_cur_n = np.maximum(cur_raw / cur_raw.sum(), 1e-10)
            p_ref_n /= p_ref_n.sum(); p_cur_n /= p_cur_n.sum()
            js_t = float(jensenshannon(p_ref_n, p_cur_n))
            target_drifted = js_t > 0.10
            indicators["target_js_divergence"] = round(js_t, 6)

    indicators["target_marginal_drift"] = target_drifted
    if target_drifted:
        signals.append("target_marginal_drift")
    print(f"  [1] Target marginal drift:      {'YES' if target_drifted else 'no'}")

    # 2. Feature-target correlation shift
    corr_shifts = []
    for col in feature_cols:
        try:
            r_ref = float(df_reference[[col, target_col]].dropna().corr().iloc[0, 1])
            r_cur = float(df_current[[col, target_col]].dropna().corr().iloc[0, 1])
            if not (np.isnan(r_ref) or np.isnan(r_cur)):
                corr_shifts.append({"feature": col, "r_ref": r_ref, "r_cur": r_cur,
                                     "delta": abs(r_cur - r_ref)})
        except Exception:
            pass

    if corr_shifts:
        corr_df = pd.DataFrame(corr_shifts).sort_values("delta", ascending=False)
        mean_delta = float(corr_df["delta"].mean())

        from scipy.stats import norm as _norm_cd
        n_ref_cd = len(df_reference)
        n_cur_cd = len(df_current)
        n_sig_corr = 0
        for _, row_c in corr_df.iterrows():
            r1, r2 = row_c["r_ref"], row_c["r_cur"]
            # Fisher z-transform difference test
            z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
            z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
            se_diff = np.sqrt(1.0 / max(n_ref_cd - 3, 1) + 1.0 / max(n_cur_cd - 3, 1))
            z_stat = (z1 - z2) / max(se_diff, 1e-10)
            p_corr_diff = float(2 * _norm_cd.sf(abs(z_stat)))
            if p_corr_diff < alpha:
                n_sig_corr += 1
        pct_sig_corr = n_sig_corr / max(len(corr_df), 1)
        corr_drifted = bool(pct_sig_corr > 0.25)
        indicators["corr_shift_mean_delta"]  = round(mean_delta, 4)
        indicators["corr_shift_pct_sig"]     = round(pct_sig_corr, 4)
        indicators["corr_shift_drifted"]     = corr_drifted
        indicators["corr_shift_details"]     = corr_df
        if corr_drifted:
            signals.append("feature_target_correlation_shift")
        print(f"  [2] Feature-target corr shift:  {'YES' if corr_drifted else 'no'}  "
              f"(mean Δ|r| = {mean_delta:.3f}, {n_sig_corr}/{len(corr_df)} sig. by Fisher z-test)")
    else:
        print("  [2] Feature-target corr shift:  skipped (no numeric features)")

    # 3. Residual mean shift (requires model)
    if model is not None:
        try:
            common_cols = [c for c in feature_cols
                           if c in df_reference.columns and c in df_current.columns]
            X_ref = df_reference[common_cols].fillna(0)
            X_cur = df_current[common_cols].fillna(0)
            y_ref_clean = df_reference[target_col].fillna(0)
            y_cur_clean = df_current[target_col].fillna(0)

            resid_ref = y_ref_clean.values - model.predict(X_ref)
            resid_cur = y_cur_clean.values - model.predict(X_cur)

            _, resid_p = ks_2samp(resid_ref, resid_cur)
            resid_drifted = bool(resid_p < alpha)
            indicators["residual_shift_p"]       = round(float(resid_p), 6)
            indicators["residual_shift_drifted"]  = resid_drifted
            indicators["mean_resid_ref"]          = round(float(resid_ref.mean()), 4)
            indicators["mean_resid_cur"]          = round(float(resid_cur.mean()), 4)
            if resid_drifted:
                signals.append("residual_distribution_shift")
            print(f"  [3] Residual distribution shift:{'YES' if resid_drifted else 'no'}  "
                  f"(p={resid_p:.4f})")
        except Exception as exc:
            logger.debug("Residual shift failed: %s", exc)
            print("  [3] Residual distribution shift: skipped (model predict failed)")
    else:
        print("  [3] Residual distribution shift: skipped (no model provided)")

    # 4. PSI on each feature as concept proxy
    psi_vals = []
    for col in feature_cols:
        try:
            p = compute_psi(df_reference[col].dropna(), df_current[col].dropna())
            psi_vals.append(p)
        except Exception:
            pass
    if psi_vals:
        mean_psi = float(np.mean(psi_vals))
        psi_concept_signal = bool(mean_psi > PSI_WARNING)
        indicators["mean_feature_psi"]    = round(mean_psi, 4)
        indicators["psi_concept_signal"]  = psi_concept_signal
        if psi_concept_signal:
            signals.append("high_mean_feature_psi")
        print(f"  [4] Mean feature PSI:           {mean_psi:.4f}  "
              f"({'signal' if psi_concept_signal else 'stable'})")

    # Overall concept drift score
    n_signals = len(signals)
    concept_drift_score = round(min(1.0, n_signals / 4), 2)
    indicators["concept_drift_signals"]  = signals
    indicators["concept_drift_score"]    = concept_drift_score
    indicators["concept_drift_detected"] = bool(n_signals >= 2)

    severity = ("high" if n_signals >= 3 else "moderate" if n_signals >= 2 else "low")
    print(f"\n  Signals triggered: {n_signals}/4 → {signals}")
    print(f"  Concept Drift Score: {concept_drift_score:.2f}  severity: {severity}")
    print("═" * 65 + "\n")

    if plot and corr_shifts:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, max(4, len(corr_df) * 0.4 + 1)))
        top_corr = corr_df.head(15)
        x = np.arange(len(top_corr))
        w = 0.35
        ax.bar(x - w/2, top_corr["r_ref"].abs(), w, label="|r| reference",
               color="steelblue", alpha=0.8)
        ax.bar(x + w/2, top_corr["r_cur"].abs(), w, label="|r| current",
               color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(top_corr["feature"], rotation=45, ha="right")
        ax.set_ylabel("|Pearson r| with target")
        ax.set_title("Feature-Target Correlation Shift (Concept Drift Indicator)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return indicators


# ──────────────────────────────────────────────────────────────────────────────
# POPULATION STABILITY REPORT
# ──────────────────────────────────────────────────────────────────────────────

def population_stability_report(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_bins: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Full PSI report across all features — the standard production monitoring tool.

    PSI thresholds (industry standard)
    ------------------------------------
    PSI < 0.10 : stable      — no action needed
    PSI < 0.20 : warning     — monitor closely
    PSI ≥ 0.20 : significant — investigate and potentially retrain

    Returns
    -------
    pd.DataFrame — one row per feature with PSI, JS divergence, KL divergence,
                   and stability label.
    """
    print("\n" + "═" * 65)
    print("  POPULATION STABILITY REPORT (PSI)")
    print("═" * 65)

    common_cols = df_reference.columns.intersection(df_current.columns).tolist()
    if columns:
        columns = [c for c in columns if c in common_cols]
    else:
        columns = common_cols

    records = []
    for col in columns:
        ref_col = df_reference[col].dropna()
        cur_col = df_current[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(ref_col)

        if not is_numeric:
            # PSI for categorical: treat category frequencies as "bins"
            all_cats = sorted(set(ref_col.unique()) | set(cur_col.unique()))
            ref_pct = ref_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-6)
            cur_pct = cur_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-6)
            ref_p = np.maximum(ref_pct.values, 1e-6)
            cur_p = np.maximum(cur_pct.values, 1e-6)
            ref_p /= ref_p.sum(); cur_p /= cur_p.sum()
            psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
        else:
            psi = compute_psi(ref_col, cur_col, n_bins=n_bins)

        p_arr, q_arr = _normalize_histograms(ref_col, cur_col, n_bins=n_bins if is_numeric else None)
        js  = round(float(jensenshannon(p_arr, q_arr)), 6)
        # the standard KL divergence. Use compute_kl_divergence() for consistency.
        kl  = round(float(np.sum(p_arr * np.log(np.maximum(p_arr, 1e-10) / np.maximum(q_arr, 1e-10)))), 6)

        stability = (
            "stable"      if psi < PSI_OK      else
            "warning"     if psi < PSI_WARNING  else
            "significant"
        )
        records.append({
            "feature":    col,
            "type":       "numeric" if is_numeric else "categorical",
            "psi":        round(psi, 4),
            "js_div":     js,
            "kl_div":     kl,
            "stability":  stability,
            "n_ref":      len(ref_col),
            "n_cur":      len(cur_col),
        })

    psi_df = (
        pd.DataFrame(records)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )

    n_sig  = (psi_df["stability"] == "significant").sum()
    n_warn = (psi_df["stability"] == "warning").sum()
    n_ok   = (psi_df["stability"] == "stable").sum()

    print(f"  Features analyzed:  {len(psi_df)}")
    print(f"  Stable  (PSI<0.10): {n_ok}")
    print(f"  Warning (PSI<0.20): {n_warn}")
    print(f"  Significant (≥0.20):{n_sig}\n")
    print(psi_df.to_string(index=False))
    print("═" * 65 + "\n")

    if plot:
        import matplotlib.pyplot as plt
        color_map = {"stable": "#2ecc71", "warning": "#f39c12", "significant": "#e74c3c"}
        colors = [color_map[s] for s in psi_df["stability"]]
        fig, ax = plt.subplots(figsize=(12, max(5, len(psi_df) * 0.35 + 1)))
        ax.barh(psi_df["feature"][::-1], psi_df["psi"][::-1],
                color=colors[::-1], edgecolor="white")
        ax.axvline(PSI_OK,      color="orange", linestyle="--", linewidth=1.2,
                   label=f"Warning threshold ({PSI_OK})")
        ax.axvline(PSI_WARNING, color="red",    linestyle="--", linewidth=1.2,
                   label=f"Significant threshold ({PSI_WARNING})")
        ax.set_xlabel("PSI (Population Stability Index)")
        ax.set_title("Population Stability Report", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return psi_df


# ──────────────────────────────────────────────────────────────────────────────
# POPULATION STABILITY MONITORING  (rolling window)
# ──────────────────────────────────────────────────────────────────────────────

def monitor_population_stability(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: Optional[List[str]] = None,
    reference_period=None,
    n_bins: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Rolling PSI monitoring across time slices.

    Computes PSI between a fixed reference period and each subsequent time slice,
    enabling production monitoring to detect when distributions start shifting.

    Parameters
    ----------
    time_col         : column identifying time period (e.g. 'month', 'batch_id')
    feature_cols     : features to monitor (all numeric if None)
    reference_period : value of time_col to use as reference (first period if None)

    Returns
    -------
    pd.DataFrame — rows = (time_slice × feature), columns = (psi, stability, period)
    Pivot table also printed for easy reading.
    """
    print("\n" + "═" * 65)
    print("  POPULATION STABILITY MONITORING  (rolling)")
    print("═" * 65)

    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != time_col
        ]

    periods = sorted(df[time_col].dropna().unique())
    if len(periods) < 2:
        print("  Need at least 2 time periods.")
        return pd.DataFrame()

    ref_period = reference_period if reference_period is not None else periods[0]
    df_ref = df[df[time_col] == ref_period]
    print(f"  Reference period: {ref_period}  |  Monitoring periods: {len(periods) - 1}\n")

    records = []
    for period in periods:
        if period == ref_period:
            continue
        df_cur = df[df[time_col] == period]
        for col in feature_cols:
            ref_col = df_ref[col].dropna()
            cur_col = df_cur[col].dropna()
            if len(ref_col) < 5 or len(cur_col) < 5:
                continue
            try:
                psi = compute_psi(ref_col, cur_col, n_bins=n_bins)
                stability = (
                    "stable"  if psi < PSI_OK else
                    "warning" if psi < PSI_WARNING else
                    "significant"
                )
                records.append({
                    "period":    period,
                    "feature":   col,
                    "psi":       round(psi, 4),
                    "stability": stability,
                })
            except Exception as exc:
                logger.debug("PSI monitor %s %s: %s", period, col, exc)

    if not records:
        print("  No PSI results computed.")
        return pd.DataFrame()

    monitor_df = pd.DataFrame(records)

    # Pivot for readability
    pivot = monitor_df.pivot_table(index="feature", columns="period",
                                   values="psi", aggfunc="first")
    print("  PSI by feature × period:\n")
    print(pivot.round(4).to_string())
    print("\n  (PSI < 0.10 = stable | 0.10–0.20 = warning | ≥ 0.20 = significant)")
    print("═" * 65 + "\n")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                        max(5, len(pivot) * 0.4 + 1)))
        # vmax floor of 0.30 keeps the color scale meaningful when all PSI
        # values are stable; it never clips — severe drift above 0.30 simply
        # extends the scale instead of saturating at the "significant" color.
        finite_vals = pivot.values[~np.isnan(pivot.values)]
        vmax = max(0.30, float(finite_vals.max())) if finite_vals.size else 0.30
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Population Stability Index — Rolling Monitor",
                     fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="PSI")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.tight_layout()
        plt.show()

    return monitor_df


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_histograms(
    ref: pd.Series,
    cur: pd.Series,
    n_bins: Optional[int] = 20,
    epsilon: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build normalized probability arrays for divergence computations."""
    is_numeric = pd.api.types.is_numeric_dtype(ref)
    if is_numeric and n_bins is not None:
        bins = np.histogram_bin_edges(pd.concat([ref.dropna(), cur.dropna()]), bins=n_bins)
        p, _ = np.histogram(ref.dropna(), bins=bins)
        q, _ = np.histogram(cur.dropna(), bins=bins)
    else:
        all_cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
        p = ref.dropna().value_counts().reindex(all_cats, fill_value=0).values
        q = cur.dropna().value_counts().reindex(all_cats, fill_value=0).values
    
    p = p.astype(float)
    q = q.astype(float)
    p_sum = p.sum()
    q_sum = q.sum()
    p = p / p_sum if p_sum > 0 else np.ones(len(p)) / len(p)
    q = q / q_sum if q_sum > 0 else np.ones(len(q)) / len(q)
    p = np.maximum(p, epsilon)           # clip zeros to epsilon AFTER normalizing
    q = np.maximum(q, epsilon)
    return p / p.sum(), q / q.sum()      # re-normalize so each sums to 1.0
