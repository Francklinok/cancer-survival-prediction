"""
statistical_analysis.py — Formal Statistical Analysis Module.

Principle
---------
This module contains ALL hypothesis tests, p-values, and statistical decisions.

Structure
---------
  1.  Normality testing               → normality_tests()
  2.  Group comparison                → compare_groups()
        - Student t-test  (2 groups, normal, equal var)
        - Welch t-test    (2 groups, normal, unequal var)
        - Mann-Whitney U  (non-parametric, 2 groups)
        - One-way ANOVA   (3+ groups, normal, equal var)
        - Welch ANOVA     (3+ groups, normal, unequal var) ← via alexandergovern
        - Kruskal-Wallis  (non-parametric, 3+ groups)
  3.  Numeric correlation             → correlation_analysis()
        - Pearson (linear, normal data)
        - Spearman (monotonic, robust)
  4.  Categorical association         → categorical_association_tests()
        - Chi-square test
        - Cramér's V (effect size)
  5.  Variance analysis               → variance_analysis()
        - Per-feature variance for preliminary feature selection
        - Levene's test for equality of variances across groups
  6.  Multicollinearity               → multicollinearity_analysis()
        - VIF (Variance Inflation Factor)
        - Correlation matrix with p-values
  7.  Effect sizes                    → effect_size_analysis()
        - Cohen's d (2-group, continuous)
        - Eta-squared η² (ANOVA / Kruskal-Wallis)
        - Cramér's V (categorical)
  8.  Feature association tests       → feature_association_tests()
        - Unified feature × target formal test (all types)
  9.  Full orchestrator               → run_statistical_analysis()

Usage
-----
    from ml_framework.analysis.statistical_analysis import run_statistical_analysis
    stats_report = run_statistical_analysis(df, target_col="TreatmentResponse")

    # Or individual analyses:
    from ml_framework.analysis.statistical_analysis import (
        normality_tests, compare_groups, correlation_analysis,
        categorical_association_tests, variance_analysis,
        multicollinearity_analysis, effect_size_analysis,
    )
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    alexandergovern,
    anderson,
    chi2_contingency,
    f_oneway,
    kruskal,
    kstest,
    levene,
    mannwhitneyu,
    pearsonr,
    shapiro,
    spearmanr,
    ttest_ind,
)

from ml_framework.utils.display_utils import section_header
from ml_framework.utils.statistical_utils import (
    classify_effect_magnitude,
    compute_cohens_d,
    compute_cohens_d as _cohens_d,
    compute_cramers_v,
    compute_cramers_v_bc,
    compute_eta_squared,
    compute_omega_squared,
    compute_rank_biserial,
    compute_vif_matrix,
    significance_stars as _stars,
    all_groups_parametric,
    choose_group_test,
    corr_with_ci,
    is_parametric_appropriate as _is_normal,
)
from ml_framework.analysis.column_analysis import analyze_column_properties



logger = logging.getLogger("ml_framework.statistical_analysis")


# =============================================================================
# 1. NORMALITY TESTING
# =============================================================================

def normality_tests(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    sw_sample_size: int = 5_000,
) -> pd.DataFrame:
    """
    Formal normality tests with statistical interpretation.

    Tests applied per column
    ------------------------
    1. Shapiro-Wilk       — best power for n ≤ 5 000; stratified sample above
    2. Anderson-Darling   — tail-sensitive, no sample size limit
    3. Kolmogorov-Smirnov — one-sample vs N(μ, σ)

    Normality score (0–1)
    ---------------------
    Each test contributes 1/3 when it passes (H₀ not rejected).
    Skewness/kurtosis penalties fine-tune borderline cases.

    Verdict bands
    -------------
    ≥ 0.75  → Normal       — parametric tests safe
    0.40–0.75 → Borderline — inspect; light transform may help
    < 0.40  → Non-normal   — use non-parametric tests

    Statistical note on large n
    --------------------------------
    For n > 5 000, Shapiro-Wilk always rejects H₀ (p ≈ 0) because its power
    grows with n and detects trivially small deviations.  At large n, rely on
    skewness, kurtosis, and Anderson-Darling instead.

    Parameters
    ----------
    df             : DataFrame
    columns        : numeric columns to test (all numeric > 2 unique if None)
    alpha          : significance threshold (default 0.05)
    sw_sample_size : max Shapiro-Wilk sample size (default 5 000)

    Returns
    -------
    pd.DataFrame indexed by column name with columns:
        n, skewness, kurtosis,
        shapiro_stat, shapiro_p, shapiro_pass,
        ad_stat, ad_critical_5pct, ad_pass,
        ks_stat, ks_p, ks_pass,
        normality_score, normality_verdict,
        recommendation
    """

    if columns is None:
        columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].nunique() > 2
        ]
    if not columns:
        logger.warning("No numeric columns for normality tests.")
        return pd.DataFrame()

    section_header("NORMALITY TESTS")
    print(f"  Tests: Shapiro-Wilk + Anderson-Darling + KS one-sample | α = {alpha}")
    print(f"  Score ≥ 0.75 → Normal | 0.40–0.75 → Borderline | < 0.40 → Non-normal\n")

    analysis = analyze_column_properties(
        df, columns=columns, alpha=alpha,
        sw_sample_size=sw_sample_size, verbose=False,
    )

    if analysis.empty:
        return pd.DataFrame()

    verdict_icon = {"Normal": "✅", "Borderline": "🟡", "Non-normal": "❌"}
    large_n = len(df) > sw_sample_size

    for _, row in analysis.iterrows():
        col     = row["column"]
        verdict = row["normality_verdict"]
        score   = row["normality_score"]
        sk      = row["skewness"]
        ku      = row["kurtosis"]
        sw_p    = row["shapiro_p"]
        ks_p    = row["ks_p"]
        n       = row["n"]
        icon    = verdict_icon.get(verdict, "?")
        sw_note = " n>5000: SW unreliable" if large_n else ""

        print(f"  {icon} {col:<28} {verdict:<12} score={score:.2f}")
        print(f"     skew={sk:+.2f}  kurt={ku:+.2f}  "
              f"SW p={'N/A' if pd.isna(sw_p) else f'{sw_p:.4f}'}{sw_note}  "
              f"KS p={'N/A' if pd.isna(ks_p) else f'{ks_p:.4f}'}")
        print(f"     → {row['linear_model_rec']}")

    result = analysis.set_index("column").rename_axis("variable")
    result["clt_applies"] = analysis["n"].gt(30).values

    print(f"\n  Summary: {(result['normality_verdict']=='Normal').sum()} Normal | "
          f"{(result['normality_verdict']=='Borderline').sum()} Borderline | "
          f"{(result['normality_verdict']=='Non-normal').sum()} Non-normal")

    if large_n:
        print(f"\n n={len(df):,} > {sw_sample_size}: Shapiro-Wilk p-values are unreliable.")
        print("     Use skewness + Anderson-Darling as primary normality indicators.")

    return result


# =============================================================================
# 2. GROUP COMPARISON TESTS
# =============================================================================

def compare_groups(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    alpha: float = 0.05,
    bonferroni: bool = True,
    equal_var: bool = False,
) -> pd.DataFrame:
    """
    Formal group comparison tests — auto-selects method by group count and normality.

    Decision logic (per feature)
    ----------------------------
    2 groups + all normal + equal var   → Student t-test
    2 groups + all normal + unequal var → Welch t-test          (default)
    2 groups + non-normal               → Mann-Whitney U
    3+ groups + all normal + equal var  → One-way ANOVA
    3+ groups + all normal + unequal    → Welch ANOVA            (new)
    3+ groups + non-normal              → Kruskal-Wallis

    Normality is assessed with a composite score (Shapiro-Wilk + Anderson-Darling
    + skewness + kurtosis) via all_groups_parametric().

    Statistical note on CLT
    -----------------------
    The CLT guarantees that *test statistics* (t, F) converge in distribution
    under large n. It does NOT imply that the *data* are Gaussian. Income data
    with n=100 000 is still log-normal. Treating large-n as a free pass to
    parametric tests ignores distributional shape and leads to biased effect
    size estimates (Cohen's d on skewed data, for example). This module does
    not apply any such shortcut — normality is assessed on the data itself.

    Multiple-testing correction
    ---------------------------
    With bonferroni=True (default), p-values are Bonferroni-corrected across
    all features tested, controlling family-wise error rate.

    Effect sizes computed
    ---------------------
    2 groups → Cohen's d
    3+ groups → η² (eta-squared) from H-statistic (Kruskal) or F (ANOVA)

    Parameters
    ----------
    df           : DataFrame
    feature_cols : numeric columns to compare across groups
    group_col    : categorical column defining groups
    alpha        : significance threshold (after correction if bonferroni=True)
    bonferroni   : apply Bonferroni correction (default True)
    equal_var    : assume equal variance for t-test (default False = Welch)

    Returns
    -------
    pd.DataFrame — one row per feature with columns:
        feature, n_groups, test_used, statistic, p_raw, p_corrected,
        significant, effect_size, effect_size_type, effect_magnitude,
        interpretation
    """
    section_header(f"GROUP COMPARISON TESTS — groups: '{group_col}'")

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    groups_vals = sorted(df[group_col].dropna().unique(), key=str)
    n_groups    = len(groups_vals)
    print(f"  Groups ({n_groups}): {list(groups_vals)}")
    print(f"  Bonferroni correction: {'ON' if bonferroni else 'OFF'} | α = {alpha}\n")

    rows = []
    p_raws = []

    for col in feature_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        group_arrays = [
            df.loc[df[group_col] == g, col].dropna().values
            for g in groups_vals
        ]
        group_arrays = [g for g in group_arrays if len(g) >= 5]

        if len(group_arrays) < 2:
            continue

        test_name, _ = choose_group_test(group_arrays, alpha=alpha)

        stat     = np.nan
        p_raw    = np.nan
        test_used = ""
        effect_size      = np.nan
        effect_size_type = ""

        try:
            if test_name == "student_t":
                a, b = group_arrays[0], group_arrays[1]
                stat, p_raw      = ttest_ind(a, b, equal_var=True)
                test_used        = "Student t-test"
                effect_size      = abs(compute_cohens_d(a, b, variant="pooled"))
                effect_size_type = "Cohen's d"

            elif test_name == "welch_t":
                a, b = group_arrays[0], group_arrays[1]
                stat, p_raw      = ttest_ind(a, b, equal_var=False)
                test_used        = "Welch t-test"
                effect_size      = abs(compute_cohens_d(a, b, variant="glass"))
                effect_size_type = "Glass Δ"

            elif test_name == "mann_whitney":
                a, b = group_arrays[0], group_arrays[1]
                stat, p_raw      = mannwhitneyu(a, b, alternative="two-sided")
                test_used        = "Mann-Whitney U"
                effect_size      = compute_rank_biserial(stat, len(a), len(b))
                effect_size_type = "rank-biserial r"

            elif test_name == "anova":
                n_tot = sum(len(g) for g in group_arrays)
                k     = len(group_arrays)
                stat, p_raw      = f_oneway(*group_arrays)
                test_used        = "One-way ANOVA"
                effect_size      = compute_omega_squared(stat, n_tot, k, test_type="anova")
                effect_size_type = "ω² (ANOVA)"

            elif test_name == "welch_anova":
                n_tot = sum(len(g) for g in group_arrays)
                k     = len(group_arrays)
                result_ag        = alexandergovern(*group_arrays)
                stat, p_raw      = float(result_ag.statistic), float(result_ag.pvalue)
                test_used        = "Welch ANOVA"
                effect_size      = compute_eta_squared(stat, n_tot, k, test_type="anova")
                effect_size_type = "η² (anova)"

            elif test_name == "kruskal":
                n_tot = sum(len(g) for g in group_arrays)
                k     = len(group_arrays)
                stat, p_raw      = kruskal(*group_arrays)
                test_used        = "Kruskal-Wallis"
                effect_size      = compute_eta_squared(stat, n_tot, k, test_type="kruskal")
                effect_size_type = "η² (kw)" 

        except Exception as exc:
            logger.warning("Test failed for '%s' (test=%s): %s", col, test_name, exc)
            continue

        p_raws.append(p_raw)
        rows.append({
            "feature":          col,
            "n_groups":         n_groups,
            "test_used":        test_used,
            "statistic":        round(float(stat), 4),
            "p_raw":            round(float(p_raw), 6),
            "p_corrected":      None,  
            "significant":      None,
            "effect_size":      round(float(effect_size), 4),
            "effect_size_type": effect_size_type,
            "effect_magnitude": _effect_magnitude(effect_size_type, effect_size),
            "interpretation":   "",
        })

    if not rows:
        print("  No valid tests computed.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    if bonferroni and len(p_raws) > 1:
        p_corrected = np.minimum(np.array(p_raws) * len(p_raws), 1.0)
    else:
        p_corrected = np.array(p_raws)

    result_df["p_corrected"] = p_corrected.round(6)
    result_df["significant"] = result_df["p_corrected"] < alpha
    result_df["interpretation"] = result_df.apply(
        lambda r: _interpret_group_test(r, alpha), axis=1
    )
    result_df = result_df.sort_values("p_corrected")

    W = 90
    print(f"  {'Feature':<25} {'Test':<18} {'Stat':>8} {'p_raw':>9} "
          f"{'p_adj':>9} {'Sig':>4} {'Effect':>8} {'Magnitude':<12}")
    print("  " + "─" * W)
    for _, r in result_df.iterrows():
        sig_star = _stars(r["p_corrected"])
        print(
            f"  {r['feature']:<25} {r['test_used']:<18} {r['statistic']:>8.3f} "
            f"{r['p_raw']:>9.4f} {r['p_corrected']:>9.4f} {sig_star:>4} "
            f"{r['effect_size']:>8.3f} {r['effect_magnitude']:<12}"
        )

    n_sig = result_df["significant"].sum()
    print(f"\n  Significant (p_adj < {alpha}): {n_sig} / {len(result_df)} features")
    if bonferroni:
        print(f"  Bonferroni correction applied across {len(result_df)} tests.")

    return result_df


def _effect_magnitude(effect_type: str, value: float) -> str:
    return classify_effect_magnitude(value, effect_type)


def _interpret_group_test(row: pd.Series, alpha: float) -> str:
    if not row["significant"]:
        return f"No significant difference across groups (p={row['p_corrected']:.4f})"
    return (
        f"Significant difference detected ({row['test_used']}, "
        f"p={row['p_corrected']:.4f}) — "
        f"{row['effect_size_type']}={row['effect_size']:.3f} ({row['effect_magnitude']} effect)"
    )


# =============================================================================
# 3. NUMERIC CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    method: str = "auto",
    alpha: float = 0.05,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Formal correlation analysis between numeric columns.

    Method selection
    ----------------
    "auto"    → Pearson if both variables normal, Spearman otherwise
    "pearson" → Pearson r (linear, assumes normality)
    "spearman"→ Spearman ρ (monotonic, robust, non-parametric)

    For each pair: coefficient, p-value, confidence interval, significance.
    If target_col provided: computes correlation of each feature with target only.

    Parameters
    ----------
    df         : DataFrame
    columns    : columns to analyze (all numeric if None)
    target_col : if provided, compute only feature × target correlations
    method     : 'auto' | 'pearson' | 'spearman'
    alpha      : significance threshold
    threshold  : minimum |r| to flag as noteworthy

    Returns
    -------
    pd.DataFrame with columns:
        var1, var2, method, r, p_value, significant, ci_low, ci_high,
        magnitude, interpretation
    """
    section_header(f"CORRELATION ANALYSIS — method: {method.upper()}")

    if columns is None:
        columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].nunique() > 2
        ]

    def _corr_pair(x: pd.Series, y: pd.Series, meth: str) -> Tuple[float, float, float, float]:
        return corr_with_ci(x, y, method=meth)

    _normality_cache: Dict[str, bool] = {}
    def _col_is_normal(col: str) -> bool:
        if col not in _normality_cache:
            vals = df[col].dropna().values.astype(float)
            _normality_cache[col] = _is_normal(vals, alpha=alpha)
        return _normality_cache[col]

    def _auto_method(col1: str, col2: str) -> str:
        if _col_is_normal(col1) and _col_is_normal(col2):
            return "pearson"
        return "spearman"

    pairs = []
    if target_col and target_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            print(
                f"  NOTE: target_col '{target_col}' is categorical — excluded from Pearson/Spearman.\n"
                f"  → Use categorical_association_tests() for cat×cat associations,\n"
                f"    or root_cause_analysis() / diagnostic_analysis() for num×cat effect sizes.\n"
            )
            target_col = None
        else:
            pairs = [(col, target_col) for col in columns if col != target_col]

    if not pairs:
        pairs = [(c1, c2) for c1, c2 in combinations(columns, 2)]

    rows = []
    for c1, c2 in pairs:
        meth = method if method != "auto" else _auto_method(c1, c2)
        r, p, ci_lo, ci_hi = _corr_pair(df[c1], df[c2], meth)
        if pd.isna(r):
            continue
        rows.append({
            "var1":           c1,
            "var2":           c2,
            "method":         meth,
            "r":              round(r, 4),
            "p_value":        round(p, 6),
            "significant":    p < alpha,
            "ci_low":         round(ci_lo, 4),
            "ci_high":        round(ci_hi, 4),
            "magnitude":      _corr_magnitude(abs(r)),
            "interpretation": _interpret_corr(r, p, alpha, c1, c2, meth),
        })

    if not rows:
        print("  No valid correlation pairs computed.")
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .sort_values("r", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    noteworthy = result[result["r"].abs() >= threshold]
    print(f"  {'Var1':<25} {'Var2':<25} {'Method':<9} {'r':>7} {'p':>9} {'Sig':>4} {'95% CI':<18} {'Magnitude'}")
    print("  " + "─" * 105)
    for _, r in noteworthy.iterrows():
        sig = _stars(r["p_value"])
        ci  = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"  {r['var1']:<25} {r['var2']:<25} {r['method']:<9} "
              f"{r['r']:>7.4f} {r['p_value']:>9.4f} {sig:>4} {ci:<18} {r['magnitude']}")

    n_sig = result["significant"].sum()
    n_pairs = len(result)
    print(f"\n  Total pairs: {n_pairs} | Significant (p<{alpha}): {n_sig} | "
          f"|r|≥{threshold}: {len(noteworthy)}")
    if n_pairs > 1:
        expected_fp = round(n_pairs * alpha, 1)
        print(
            f"  ⚠  No multiple-testing correction applied across {n_pairs} pairs."
            f" At α={alpha}, ~{expected_fp} false positive(s) expected by chance."
            f" Apply Bonferroni: α_adj = {alpha/n_pairs:.5f}, or use correlation_analysis()"
            f" after feature_association_tests() (which applies Bonferroni)."
        )

    return result


def _corr_magnitude(r_abs: float) -> str:
    return classify_effect_magnitude(r_abs, "pearson")


def _interpret_corr(r: float, p: float, alpha: float, c1: str, c2: str, method: str) -> str:
    direction = "positive" if r > 0 else "negative"
    mag       = _corr_magnitude(abs(r))
    sig_txt   = f"significant (p={p:.4f})" if p < alpha else f"not significant (p={p:.4f})"
    return f"{method.capitalize()} {direction} {mag} correlation between '{c1}' and '{c2}' — {sig_txt}"


# =============================================================================
# 4. CATEGORICAL ASSOCIATION TESTS
# =============================================================================

def categorical_association_tests(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    alpha: float = 0.05,
    bonferroni: bool = True,
    min_expected: float = 5.0,
) -> pd.DataFrame:
    """
    Chi-square tests + Cramér's V effect size for categorical associations.

    Chi-square assumptions
    ----------------------
    - Expected frequency ≥ 5 in all cells (checked automatically)
    - If violated: Fisher's exact test is used for 2×2, warning for larger

    Effect size: Cramér's V (bias-corrected, Bergsma 2013)
    -------------------------------------------------------
    The classic Cramér's V overestimates effect size on small samples because
    chi² > 0 even under H₀. The bias-corrected version (V_bc) is used here.
    V_bc ≈ V for n ≥ 200 but is more accurate for smaller contingency tables.

    V = 0.0–0.1  → negligible
    V = 0.1–0.3  → weak
    V = 0.3–0.5  → moderate
    V > 0.5      → strong

    Parameters
    ----------
    df           : DataFrame
    cat_cols     : categorical columns to test (all object/category if None)
    target_col   : if provided, test each cat_col × target only
    alpha        : significance threshold
    bonferroni   : Bonferroni correction across all tests
    min_expected : minimum expected cell count threshold (default 5)

    Returns
    -------
    pd.DataFrame with columns:
        var1, var2, chi2, dof, p_raw, p_corrected, significant,
        cramers_v, effect_magnitude, assumption_ok, interpretation
    """
    section_header("CATEGORICAL ASSOCIATION TESTS — Chi-Square + Cramér's V")

    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    pairs = []
    if target_col and target_col in df.columns:
        pairs = [(col, target_col) for col in cat_cols if col != target_col]
    else:
        pairs = [(c1, c2) for c1, c2 in combinations(cat_cols, 2)]

    rows   = []
    p_raws = []

    for c1, c2 in pairs:
        try:
            ct           = pd.crosstab(df[c1].dropna(), df[c2].dropna())
            chi2_val, p, dof, expected = chi2_contingency(ct)
            assume_ok    = bool((expected >= min_expected).all())
            n            = int(ct.sum().sum())
            n_rows, n_cols = ct.shape
            cramers_v    = compute_cramers_v_bc(chi2_val, n, n_rows, n_cols)

            p_raws.append(p)
            rows.append({
                "var1":             c1,
                "var2":             c2,
                "chi2":             round(chi2_val, 3),
                "dof":              dof,
                "p_raw":            round(p, 6),
                "p_corrected":      None,
                "significant":      None,
                "cramers_v":        round(cramers_v, 4),
                "effect_magnitude": _cramers_magnitude(cramers_v),
                "assumption_ok":    assume_ok,
                "n":                int(n),
                "interpretation":   "",
            })
        except Exception as exc:
            logger.warning("Chi-square failed for ('%s', '%s'): %s", c1, c2, exc)

    if not rows:
        print("  No valid tests computed.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    if bonferroni and len(p_raws) > 1:
        p_corrected = np.minimum(np.array(p_raws) * len(p_raws), 1.0)
    else:
        p_corrected = np.array(p_raws)

    result["p_corrected"] = p_corrected.round(6)
    result["significant"] = result["p_corrected"] < alpha
    result["interpretation"] = result.apply(
        lambda r: _interpret_chi2(r, alpha), axis=1
    )
    result = result.sort_values("cramers_v", ascending=False).reset_index(drop=True)

    print(f"  {'Var1':<25} {'Var2':<25} {'χ²':>8} {'dof':>4} {'p_adj':>9} "
          f"{'Sig':>4} {'V':>6} {'Magnitude':<12} {'Assum.'}")
    print("  " + "─" * 100)
    for _, r in result.iterrows():
        sig   = _stars(r["p_corrected"])
        assum = "✅" if r["assumption_ok"] else "⚠️"
        print(
            f"  {r['var1']:<25} {r['var2']:<25} {r['chi2']:>8.2f} {r['dof']:>4} "
            f"{r['p_corrected']:>9.4f} {sig:>4} {r['cramers_v']:>6.3f} "
            f"{r['effect_magnitude']:<12} {assum}"
        )

    n_sig = result["significant"].sum()
    n_assum_violated = (~result["assumption_ok"]).sum()
    print(f"\n  Significant: {n_sig} / {len(result)}")
    if n_assum_violated:
        print(f"  ⚠️  {n_assum_violated} test(s) with expected cell < {min_expected} — interpret with caution.")
    if bonferroni:
        print(f"  Bonferroni correction applied across {len(result)} tests.")

    return result


def _cramers_magnitude(v: float) -> str:
    return classify_effect_magnitude(v, "cramers_v")


def _interpret_chi2(row: pd.Series, alpha: float) -> str:
    assum = "" if row["assumption_ok"] else " ⚠️ (assumption violated)"
    if not row["significant"]:
        return f"No significant association (p={row['p_corrected']:.4f}){assum}"
    return (
        f"Significant association (χ²={row['chi2']:.2f}, p={row['p_corrected']:.4f}) — "
        f"Cramér's V={row['cramers_v']:.3f} ({row['effect_magnitude']} effect){assum}"
    )


# =============================================================================
# 5. VARIANCE ANALYSIS
# =============================================================================

def variance_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    alpha: float = 0.05,
    low_var_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Variance analysis for feature selection and group equality testing.

    Two modes
    ---------
    1. No group_col → per-feature variance ranking (preliminary feature selection)
       Flags near-zero variance features (CV < low_var_threshold).

    2. With group_col → Levene's test for equality of variances across groups
       (assumption check for t-test / ANOVA).
       Levene's test: H₀ = all groups have equal variance.

    Parameters
    ----------
    df                : DataFrame
    feature_cols      : numeric columns (all numeric if None)
    group_col         : grouping column for Levene's test (None for univariate mode)
    alpha             : significance threshold
    low_var_threshold : CV threshold for near-zero variance flag (default 0.01)

    Returns
    -------
    pd.DataFrame with variance metrics and Levene test results if group_col given
    """
    section_header("VARIANCE ANALYSIS")

    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].nunique() > 2
        ]

    rows = []
    for col in feature_cols:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        var  = float(s.var())
        std  = float(s.std())
        mean = float(s.mean())
        cv   = abs(std / mean) if abs(mean) > 1e-10 else float("inf")
        near_zero = cv < low_var_threshold or var < 1e-8

        row_data = {
            "feature":     col,
            "n":           len(s),
            "mean":        round(mean, 4),
            "std":         round(std,  4),
            "variance":    round(var,  6),
            "cv":          round(cv,   4),
            "near_zero_var": near_zero,
        }

        if group_col and group_col in df.columns:
            groups = [
                df.loc[df[group_col] == g, col].dropna().values
                for g in df[group_col].dropna().unique()
            ]
            groups = [g for g in groups if len(g) >= 2]
            levene_stat = levene_p = np.nan
            equal_var   = None
            if len(groups) >= 2:
                try:
                    levene_stat, levene_p = levene(*groups)
                    equal_var = levene_p > alpha
                except Exception:
                    pass
            row_data.update({
                "levene_stat": round(float(levene_stat), 4) if not np.isnan(levene_stat) else np.nan,
                "levene_p":    round(float(levene_p),    6) if not np.isnan(levene_p)    else np.nan,
                "equal_var":   equal_var,
            })

        rows.append(row_data)

    result = pd.DataFrame(rows).sort_values("variance", ascending=False)

    if group_col:
        print(f"  Levene's test for equal variance across '{group_col}' groups | α={alpha}\n")
        print(f"  {'Feature':<28} {'Var':>10} {'CV':>6} {'Levene F':>10} {'p':>9} {'Sig':>4} {'Equal var?'}")
        print("  " + "─" * 80)
        for _, r in result.iterrows():
            lp  = r.get("levene_p", np.nan)
            sig = _stars(lp) if not pd.isna(lp) else "—"
            eq  = "✅ yes" if r.get("equal_var") else ("❌ no" if r.get("equal_var") is False else "—")
            print(
                f"  {r['feature']:<28} {r['variance']:>10.4f} {r['cv']:>6.3f} "
                f"{r.get('levene_stat', float('nan')):>10.3f} "
                f"{lp:>9.4f} {sig:>4} {eq}"
            )
        n_unequal = (~result["equal_var"].fillna(True)).sum()
        if n_unequal:
            print(f"\n  ⚠️  {n_unequal} feature(s) with unequal variance → use Welch t-test or non-parametric test.")
    else:
        print(f"  Variance ranking — low_var threshold: CV < {low_var_threshold}\n")
        print(f"  {'Feature':<28} {'Variance':>10} {'Std':>8} {'CV':>7} {'Near-zero?'}")
        print("  " + "─" * 65)
        for _, r in result.iterrows():
            flag = "⚠️ drop candidate" if r["near_zero_var"] else ""
            print(f"  {r['feature']:<28} {r['variance']:>10.4f} {r['std']:>8.4f} {r['cv']:>7.3f}  {flag}")
        n_low = result["near_zero_var"].sum()
        if n_low:
            print(f"\n  ⚠️  {n_low} near-zero variance feature(s) — likely uninformative for ML.")

    return result


# =============================================================================
# 6. MULTICOLLINEARITY — VIF
# =============================================================================

def multicollinearity_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    vif_threshold: float = 5.0,
    corr_threshold: float = 0.80,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Multicollinearity analysis — VIF + Pearson correlation matrix with p-values.

    VIF interpretation
    ------------------
    VIF = 1         → no collinearity
    VIF = 1–5       → moderate (acceptable)
    VIF = 5–10      → high — consider removing or combining
    VIF > 10        → severe — remove or apply PCA / Ridge regularization

    Also returns all pairs with |r| > corr_threshold and their p-values.

    Parameters
    ----------
    df             : DataFrame
    feature_cols   : numeric columns (all numeric if None)
    vif_threshold  : VIF flag threshold (default 5.0)
    corr_threshold : Pearson |r| threshold for high-correlation flag (default 0.80)
    alpha          : p-value threshold for correlation significance

    Returns
    -------
    Tuple of two DataFrames: (vif_df, high_corr_df)
    vif_df       : VIF per feature
    high_corr_df : pairs with |r| > corr_threshold + p-values
    """
    section_header("MULTICOLLINEARITY ANALYSIS — VIF + Correlation")

    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].nunique() > 2
        ]

    valid_cols = [c for c in feature_cols
                  if c in df.columns and df[c].dropna().std() > 1e-8]
    X = df[valid_cols].dropna()
    n_dropped = len(df) - len(X)
    if n_dropped > 0:
        pct_dropped = n_dropped / len(df) * 100
        msg = f"VIF/correlation: {n_dropped} rows ({pct_dropped:.1f}%) dropped (listwise deletion)."
        if pct_dropped > 20:
            logger.warning(
                "%s If missing data is MNAR, VIF estimates may be biased. "
                "Consider imputing before calling multicollinearity_analysis().", msg
            )
        else:
            logger.info("%s", msg)

    if len(valid_cols) < 2:
        print("  Not enough valid columns for multicollinearity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        vif_df = compute_vif_matrix(X, columns=valid_cols, threshold=vif_threshold)
        vif_df = vif_df.rename(columns={"vif": "vif", "severity": "severity"})
    except Exception as exc:
        logger.warning("VIF computation failed: %s", exc)
        vif_df = pd.DataFrame()

    print(f"  VIF threshold: {vif_threshold} | Features: {len(valid_cols)}\n")
    print(f"  {'Feature':<30} {'VIF':>8} {'R²(others)':>11} {'Severity'}")
    print("  " + "─" * 58)
    for _, r in vif_df.iterrows():
        flag = " HIGH" if r["vif"] >= vif_threshold else ""
        print(f"  {r['feature']:<30} {r['vif']:>8.2f} {r['r2_others']:>11.4f}  {r['severity']} {flag}")

    n_high = (vif_df["vif"] >= vif_threshold).sum()
    if n_high:
        print(f"\n {n_high} feature(s) with VIF ≥ {vif_threshold} — multicollinearity risk.")
        print("     → Remove one from each collinear pair, or apply PCA / Ridge regression.")

    print(f"\n  High-correlation pairs (|r| ≥ {corr_threshold}, Pearson + p-value):\n")
    corr_rows = []
    for c1, c2 in combinations(valid_cols, 2):
        try:
            common = X[c1].notna() & X[c2].notna()
            r, p = pearsonr(X.loc[common, c1], X.loc[common, c2])
            if abs(r) >= corr_threshold:
                corr_rows.append({
                    "var1":      c1,
                    "var2":      c2,
                    "pearson_r": round(r, 4),
                    "p_value":   round(p, 6),
                    "significant": p < alpha,
                    "action":    "Remove one or apply PCA" if abs(r) > 0.90 else "Monitor",
                })
        except Exception:
            pass

    if corr_rows:
        high_corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r", key=abs, ascending=False)
        print(f"  {'Var1':<25} {'Var2':<25} {'r':>7} {'p':>9} {'Sig':>4} {'Action'}")
        print("  " + "─" * 80)
        for _, r in high_corr_df.iterrows():
            sig = _stars(r["p_value"])
            print(f"  {r['var1']:<25} {r['var2']:<25} {r['pearson_r']:>7.4f} "
                  f"{r['p_value']:>9.4f} {sig:>4} {r['action']}")
    else:
        high_corr_df = pd.DataFrame()
        print(f" No pairs with |r| ≥ {corr_threshold}")

    return vif_df, high_corr_df


# =============================================================================
# 7. EFFECT SIZES
# =============================================================================

def effect_size_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute effect sizes for all numeric features vs. a grouping variable.

    Effect sizes computed
    ---------------------
    2 groups  → Cohen's d  (standardized mean difference)
    3+ groups → η² eta-squared (proportion of variance explained by group)
                Cramér's V for categorical features

    Cohen's d interpretation (conventional)
    ----------------------------------------
    |d| < 0.20  → negligible
    |d| < 0.50  → small
    |d| < 0.80  → medium
    |d| ≥ 0.80  → large

    η² interpretation
    -----------------
    η² < 0.01   → negligible
    η² < 0.06   → small
    η² < 0.14   → medium
    η² ≥ 0.14   → large

    Cramér's V interpretation
    -------------------------
    V < 0.10    → negligible
    V < 0.30    → weak
    V < 0.50    → moderate
    V ≥ 0.50    → strong

    Parameters
    ----------
    df           : DataFrame
    feature_cols : columns to analyze (numeric + categorical)
    group_col    : grouping variable
    alpha        : for significance context only

    Returns
    -------
    pd.DataFrame — one row per feature with:
        feature, effect_type, effect_size, magnitude, test_stat, p_value,
        interpretation
    """
    section_header(f"EFFECT SIZE ANALYSIS — groups: '{group_col}'")

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    groups_vals = df[group_col].dropna().unique()
    n_groups    = len(groups_vals)
    rows        = []

    for col in feature_cols:
        if col not in df.columns or col == group_col:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            group_arrays = [
                df.loc[df[group_col] == g, col].dropna().values
                for g in groups_vals
            ]
            group_arrays = [g for g in group_arrays if len(g) >= 5]
            if len(group_arrays) < 2:
                continue

            try:
                if n_groups == 2:
                    a, b = group_arrays[0], group_arrays[1]
                    t, p = ttest_ind(a, b, equal_var=False)
                    # pooled d for uniform ranking; compare_groups() switches to Glass Δ when variances differ
                    d = abs(compute_cohens_d(a, b, variant="pooled"))
                    rows.append({
                        "feature":        col,
                        "effect_type":    "Cohen's d (pooled)",
                        "effect_size":    round(d, 4),
                        "magnitude":      _effect_magnitude("cohens_d", d),
                        "test_stat":      round(float(t), 4),
                        "p_value":        round(float(p), 6),
                        "interpretation": _d_interpret(d, col, str(groups_vals[0]), str(groups_vals[1])),
                    })
                else:
                    h, p  = kruskal(*group_arrays)
                    n_tot = sum(len(g) for g in group_arrays)
                    k     = len(group_arrays)
                    eta2  = compute_eta_squared(h, n_tot, k, test_type="kruskal")
                    rows.append({
                        "feature":        col,
                        "effect_type":    "η² (KW)",
                        "effect_size":    round(eta2, 4),
                        "magnitude":      _effect_magnitude("eta_squared", eta2),
                        "test_stat":      round(float(h), 4),
                        "p_value":        round(float(p), 6),
                        "interpretation": _eta2_interpret(eta2, col),
                    })
            except Exception as exc:
                logger.warning("Effect size failed for '%s': %s", col, exc)

        elif pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype) == "category":
            try:
                ct = pd.crosstab(df[col], df[group_col])
                chi2_val, p, _, _ = chi2_contingency(ct)
                n = int(ct.sum().sum())
                n_rows_ct, n_cols_ct = ct.shape
                v = compute_cramers_v_bc(chi2_val, n, n_rows_ct, n_cols_ct)
                rows.append({
                    "feature":        col,
                    "effect_type":    "Cramér's V (bc)",
                    "effect_size":    round(v, 4),
                    "magnitude":      _cramers_magnitude(v),
                    "test_stat":      round(float(chi2_val), 4),
                    "p_value":        round(float(p), 6),
                    "interpretation": f"Cramér's V_bc={v:.3f} ({_cramers_magnitude(v)}) between '{col}' and '{group_col}'",
                })
            except Exception as exc:
                logger.warning("Cramér's V failed for '%s': %s", col, exc)

    if not rows:
        print("  No effect sizes computed.")
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("effect_size", ascending=False)

    print(f"  {'Feature':<28} {'Type':<18} {'Effect':>8} {'Magnitude':<12} {'p':>9} {'Sig':>4}")
    print("  " + "─" * 85)
    for _, r in result.iterrows():
        sig = _stars(r["p_value"])
        print(
            f"  {r['feature']:<28} {r['effect_type']:<18} {r['effect_size']:>8.4f} "
            f"{r['magnitude']:<12} {r['p_value']:>9.4f} {sig:>4}"
        )

    large = result[result["magnitude"].isin(["large", "strong"])]
    if not large.empty:
        print(f"\n  Large/strong effects ({len(large)}):")
        for _, r in large.iterrows():
            print(f"    → {r['interpretation']}")

    return result


def _d_interpret(d: float, col: str, g1: str, g2: str) -> str:
    mag = _effect_magnitude("cohens_d", d)
    return f"Cohen's d={d:.3f} ({mag}) — '{col}' differs between '{g1}' and '{g2}'"


def _eta2_interpret(eta2: float, col: str) -> str:
    mag = _effect_magnitude("η²", eta2)
    return f"η²={eta2:.3f} ({mag}) — group membership explains {eta2*100:.1f}% of '{col}' variance"


# =============================================================================
# 8. UNIFIED FEATURE ASSOCIATION TESTS
# =============================================================================

def feature_association_tests(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> pd.DataFrame:
    """
    Formal feature × target association tests — correct test per type pair.

    Test selection matrix
    ---------------------
    Numeric feature  × Numeric target   → Pearson (normal) or Spearman (robust)
    Numeric feature  × Categorical tgt  → ANOVA (normal) or Kruskal-Wallis
    Categorical feat × Categorical tgt  → Chi-square + Cramér's V
    Categorical feat × Numeric target   → Kruskal-Wallis (η²)

    Includes effect size for every test.
    Bonferroni correction applied across all feature tests.

    Returns
    -------
    pd.DataFrame — one row per feature, sorted by significance + effect size
    """
    section_header(f"FEATURE ASSOCIATION TESTS — target: '{target_col}'")

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found.")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    # nunique <= 15 treats low-cardinality numerics as categorical (e.g. score scales).
    # Verify target type if the column sits near this boundary.
    is_cat_tgt = (
        not pd.api.types.is_numeric_dtype(df[target_col])
        or df[target_col].nunique() <= 15
    )

    rows   = []
    p_raws = []

    for col in feature_cols:
        if col not in df.columns or col == target_col:
            continue

        is_num_feat = pd.api.types.is_numeric_dtype(df[col])
        stat = p = effect = np.nan
        test_used = effect_type = ""

        try:
            if is_num_feat and not is_cat_tgt:
                # Spearman (robust, no normality assumption)
                common = df[[col, target_col]].dropna()
                r, p = spearmanr(common[col], common[target_col])
                stat, effect, test_used, effect_type = r, abs(r), "Spearman ρ", "Spearman ρ"

            elif is_num_feat and is_cat_tgt:
                groups = [
                    df.loc[df[target_col] == g, col].dropna().values
                    for g in df[target_col].dropna().unique()
                ]
                groups = [g for g in groups if len(g) >= 5]
                if len(groups) >= 2:
                    h, p  = kruskal(*groups)
                    n_tot = sum(len(g) for g in groups)
                    k     = len(groups)
                    eta2  = compute_eta_squared(h, n_tot, k, test_type="kruskal")
                    stat, effect, test_used, effect_type = h, eta2, "Kruskal-Wallis", "η²"

            elif not is_num_feat and is_cat_tgt:
                ct = pd.crosstab(df[col].dropna(), df[target_col].dropna())
                chi2_val, p, _, _ = chi2_contingency(ct)
                n = int(ct.sum().sum())
                n_r, n_c = ct.shape
                v = compute_cramers_v_bc(chi2_val, n, n_r, n_c)
                stat, effect, test_used, effect_type = chi2_val, v, "Chi-square", "Cramér's V (bc)"

            else:
                groups = [
                    df.loc[df[col] == g, target_col].dropna().values
                    for g in df[col].dropna().unique()
                ]
                groups = [g for g in groups if len(g) >= 5]
                if len(groups) >= 2:
                    h, p  = kruskal(*groups)
                    n_tot = sum(len(g) for g in groups)
                    k     = len(groups)
                    eta2  = compute_eta_squared(h, n_tot, k, test_type="kruskal")
                    stat, effect, test_used, effect_type = h, eta2, "KW (cat->num)", "η²"

            if not pd.isna(p):
                p_raws.append(p)
                rows.append({
                    "feature":       col,
                    "test":          test_used,
                    "statistic":     round(float(stat),   4),
                    "p_raw":         round(float(p),      6),
                    "p_corrected":   None,
                    "significant":   None,
                    "effect_size":   round(float(effect), 4),
                    "effect_type":   effect_type,
                    "magnitude":     _effect_magnitude(effect_type, effect),
                })

        except Exception as exc:
            logger.debug("Test failed for '%s': %s", col, exc)

    if not rows:
        print("  No valid tests computed.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    if bonferroni and len(p_raws) > 1:
        p_corr = np.minimum(np.array(p_raws) * len(p_raws), 1.0)
    else:
        p_corr = np.array(p_raws)

    result["p_corrected"] = p_corr.round(6)
    result["significant"] = result["p_corrected"] < alpha
    result = result.sort_values(["significant", "effect_size"], ascending=[False, False])

    print(f"  {'Feature':<28} {'Test':<18} {'Stat':>8} {'p_adj':>9} "
          f"{'Sig':>4} {'Effect':>8} {'Type':<15} {'Magnitude'}")
    print("  " + "─" * 100)
    for _, r in result.iterrows():
        sig = _stars(r["p_corrected"])
        print(
            f"  {r['feature']:<28} {r['test']:<18} {r['statistic']:>8.3f} "
            f"{r['p_corrected']:>9.4f} {sig:>4} {r['effect_size']:>8.4f} "
            f"{r['effect_type']:<15} {r['magnitude']}"
        )

    n_sig = result["significant"].sum()
    print(f"\n  Significant features: {n_sig} / {len(result)}")
    if bonferroni:
        print(f"  Bonferroni correction applied (n={len(result)} tests).")

    return result


# =============================================================================
# 9. FULL ORCHESTRATOR
# =============================================================================

def run_statistical_analysis(
    df: pd.DataFrame,
    target_col: str,
    steps: Optional[List[str]] = None,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete formal statistical analysis pipeline.

    Steps
    -----
    "normality"      → normality_tests()
    "groups"         → compare_groups()  [requires categorical target]
    "correlation"    → correlation_analysis()
    "categorical"    → categorical_association_tests()
    "variance"       → variance_analysis()
    "multicollinearity" → multicollinearity_analysis()
    "effect_sizes"   → effect_size_analysis()
    "associations"   → feature_association_tests()

    Parameters
    ----------
    df         : clean, encoded DataFrame
    target_col : target column name
    steps      : list of steps to run (all if None)
    alpha      : significance threshold throughout
    bonferroni : apply Bonferroni correction where applicable

    Returns
    -------
    Dict[str, pd.DataFrame] — one entry per step, keyed by step name
    """
    ALL_STEPS = [
        "normality", "groups", "correlation",
        "categorical", "variance", "multicollinearity",
        "effect_sizes", "associations",
    ]
    steps = steps or ALL_STEPS

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col and df[c].nunique() > 2
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col
    ]
    all_features = num_cols + cat_cols
    is_cat_tgt = (
        not pd.api.types.is_numeric_dtype(df[target_col])
        or df[target_col].nunique() <= 15
    )

    try:
        print("\n" + "╔" + "═" * 66 + "╗")
        print("║{:^66}║".format("  STATISTICAL ANALYSIS REPORT  "))
        print("║{:^66}║".format(f"  Target: {target_col}  |  α = {alpha}  |  Bonferroni: {bonferroni}  "))
        print("║{:^66}║".format(f"  {df.shape[0]:,} rows × {df.shape[1]} columns  "))
        print("╚" + "═" * 66 + "╝\n")
    except UnicodeEncodeError:
        print("\n" + "+" + "=" * 66 + "+")
        print("|{:^66}|".format("  STATISTICAL ANALYSIS REPORT  "))
        print("|{:^66}|".format(f"  Target: {target_col}  |  alpha = {alpha}  |  Bonferroni: {bonferroni}  "))
        print("|{:^66}|".format(f"  {df.shape[0]:,} rows x {df.shape[1]} columns  "))
        print("+" + "=" * 66 + "+\n")

    results: Dict[str, pd.DataFrame] = {}

    step_map = {
        "normality": lambda: normality_tests(df, num_cols, alpha=alpha),
        "groups":    lambda: compare_groups(df, num_cols, target_col, alpha=alpha, bonferroni=bonferroni)
                             if is_cat_tgt else None,
        "correlation":    lambda: correlation_analysis(df, num_cols, target_col, alpha=alpha),
        "categorical":    lambda: categorical_association_tests(df, cat_cols, target_col, alpha=alpha, bonferroni=bonferroni),
        "variance":       lambda: variance_analysis(df, num_cols, target_col if is_cat_tgt else None, alpha=alpha),
        "multicollinearity": lambda: multicollinearity_analysis(df, num_cols, alpha=alpha),
        "effect_sizes":   lambda: effect_size_analysis(df, all_features, target_col, alpha=alpha)
                                  if is_cat_tgt else None,
        "associations":   lambda: feature_association_tests(df, target_col, all_features, alpha=alpha, bonferroni=bonferroni),
    }

    for i, step in enumerate(steps, 1):
        if step not in step_map:
            logger.warning("Unknown step: '%s'", step)
            continue
        print(f"\n{'─' * 68}")
        print(f"  [{i}/{len(steps)}]  {step.upper()}")
        print(f"{'─' * 68}")
        try:
            out = step_map[step]()
            if out is not None:
                if isinstance(out, tuple):
                    # multicollinearity returns (vif_df, corr_df)
                    results[step + "_vif"]  = out[0]
                    results[step + "_corr"] = out[1]
                else:
                    results[step] = out
        except Exception as e:
            logger.error("Error at step '%s': %s", step, e, exc_info=True)

    print("\n" + "═" * 68)
    print("  ✅ Statistical analysis complete.")
    print("     Results stored in returned dict — access by step name.")
    print("═" * 68 + "\n")

    return results
