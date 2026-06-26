"""
statistical_utils.py — Shared statistical primitives for the ml_framework.

All functions here are PURE (no side effects, no print, no plot).
They are called by analysis/, features/, and diagnostic/ modules to avoid
duplicating the same calculation logic in multiple places.

Design constraints
------------------
- No function returns a "decision" based on a single criterion.
- Normality assessment uses a composite score (SW + AD + skewness + kurtosis),
  not a single p-value threshold.
- CLT is never used as a proxy for normality of the *data distribution*.
  It is used only to document that test *statistics* converge under large n —
  which is a separate claim from X ~ N(μ, σ).
- Levene's test is only run when normality is confirmed (it is irrelevant
  for non-parametric test selection).
- Effect sizes are computed with the estimator that matches the test used.

Functions
---------
  Normality assessment
    normality_score(arr, alpha, sw_max_n)       → NormalityResult (namedtuple)
    is_parametric_appropriate(arr, alpha, ...)  → bool
    all_groups_parametric(groups, ...)           → bool

  Test selection
    choose_group_test(groups, alpha)            → (test_name, diagnostics_dict)
    choose_correlation_method(x, y)             → 'pearson' | 'spearman'

  Effect sizes
    compute_cramers_v_bc(chi2, n, r, c)        → float  (bias-corrected, Bergsma 2013)
    compute_cramers_v(chi2_val, n, k)           → float  (classic, kept for compat.)
    compute_cohens_d(a, b, variant)             → float  (pooled | glass | hedges)
    compute_glass_delta(a, b, control)          → float
    compute_hedges_g(a, b)                      → float
    compute_eta_squared(stat, n_total, k, ...)  → float
    compute_omega_squared(stat, n_total, k, ...)→ float  (unbiased η²)
    compute_rank_biserial(U, n1, n2)            → float  (documented convention)
    classify_effect_magnitude(value, type)      → str

  Correlation helpers
    corr_with_ci(x, y, method, bootstrap_n)    → (r, p, ci_low, ci_high)
    cramers_v_from_series(x, y)                → float

  VIF
    compute_vif_series(X_df, col)              → float
    compute_vif_matrix(df, columns, threshold) → pd.DataFrame

  Significance stars
    significance_stars(p)                      → str  ('***' | '**' | '*' | 'ns')
"""

from __future__ import annotations

import logging
from collections import namedtuple
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    anderson,
    chi2_contingency,
    kurtosistest,
    levene,
    pearsonr,
    shapiro,
    skewtest,
    spearmanr,
)
from scipy.stats import norm as _norm
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("ml_framework.statistical_utils")


# =============================================================================
# EFFECT SIZE BOUNDARIES  (single source of truth)
# =============================================================================

_EFFECT_BOUNDARIES = {
    "cohens_d":      ([0.20, 0.50, 0.80],           ["negligible", "small", "medium", "large"]),
    "glass_delta":   ([0.20, 0.50, 0.80],           ["negligible", "small", "medium", "large"]),
    "hedges_g":      ([0.20, 0.50, 0.80],           ["negligible", "small", "medium", "large"]),
    "rank_biserial": ([0.10, 0.30, 0.50],           ["negligible", "small", "medium", "large"]),
    "eta_squared":   ([0.01, 0.06, 0.14],           ["negligible", "small", "medium", "large"]),
    "omega_squared": ([0.01, 0.06, 0.14],           ["negligible", "small", "medium", "large"]),
    "pearson":       ([0.10, 0.30, 0.50, 0.70],     ["negligible", "weak", "moderate", "strong", "very strong"]),
    "spearman":      ([0.10, 0.30, 0.50, 0.70],     ["negligible", "weak", "moderate", "strong", "very strong"]),
    "cramers_v":     ([0.10, 0.30, 0.50],           ["negligible", "weak", "moderate", "strong"]),
}

_EFFECT_ALIASES = {
    "cohen's d":           "cohens_d",
    "cohen's d (pooled)":  "cohens_d",
    "glass δ":             "glass_delta",
    "glass delta":         "glass_delta",
    "hedges g":            "hedges_g",
    "hedges' g":           "hedges_g",
    "rank-biserial r":     "rank_biserial",
    "η² (anova)":          "eta_squared",
    "η² (welch anova)":    "eta_squared",
    "η² (kw)":             "eta_squared",
    "η²":                  "eta_squared",
    "ω² (anova)":          "omega_squared",
    "ω²":                  "omega_squared",
    "omega²":              "omega_squared",
    "cramér's v":          "cramers_v",
    "cramér's v (bc)":     "cramers_v",
    "cramer's v":          "cramers_v",
    "spearman ρ":          "spearman",
}


def classify_effect_magnitude(value: float, effect_type: str) -> str:
    """
    Convert a numeric effect size to a human-readable magnitude label.

    Parameters
    ----------
    value       : the effect size value (sign is ignored)
    effect_type : canonical key or alias string (case-insensitive)

    Returns
    -------
    str — magnitude label (e.g. 'negligible', 'small', 'medium', 'large')
    """
    if pd.isna(value):
        return "—"
    key = _EFFECT_ALIASES.get(effect_type.lower(), effect_type.lower().replace(" ", "_"))
    config = _EFFECT_BOUNDARIES.get(key, _EFFECT_BOUNDARIES["cohens_d"])
    boundaries, labels = config
    v = abs(value)
    for threshold, label in zip(boundaries, labels):
        if v < threshold:
            return label
    return labels[-1]


# =============================================================================
# NORMALITY ASSESSMENT  (composite, multi-test)
# =============================================================================

NormalityResult = namedtuple(
    "NormalityResult",
    ["score", "is_normal", "sw_p", "ad_stat", "ad_critical_5pct",
     "skewness", "kurtosis", "n", "notes"],
)
"""
score            : float in [0, 1] — composite normality evidence.
is_normal        : bool — True if score >= threshold (default 0.60).
sw_p             : float | None — Shapiro-Wilk p-value (None if n > sw_max_n).
ad_stat          : float — Anderson-Darling statistic.
ad_critical_5pct : float — AD critical value at 5%.
skewness         : float — sample skewness.
kurtosis         : float — excess kurtosis (0 = Gaussian).
n                : int — effective sample size.
notes            : list[str] — human-readable diagnostics.
"""


def normality_score(
    arr: np.ndarray,
    alpha: float = 0.05,
    sw_max_n: int = 5_000,
    normal_threshold: float = 0.60,
) -> NormalityResult:
    """
    Composite normality assessment from multiple independent signals.

    Scoring system
    --------------
    Each component contributes to a score in [0, 1]:

    1. Shapiro-Wilk (weight 0.35)
       - Applied only when n ≤ sw_max_n. At larger n SW reliably rejects H₀
         even for trivial, practically irrelevant deviations — so it is omitted
         rather than used as a false-positive factory.

    2. Anderson-Darling (weight 0.35)
       - Applied at all n. More tail-sensitive than SW, no sample-size limit.

    3. Skewness D'Agostino test (weight 0.15)
       - |skewness| < 0.5 is a supporting signal. Applied at n ≥ 8.

    4. Kurtosis Anscombe-Glynn test (weight 0.15)
       - |excess kurtosis| < 1.0 is a supporting signal. Applied at n ≥ 20.

    IMPORTANT: a high score (is_normal=True) means the data is CONSISTENT with
    normality given these tests — NOT that the distribution IS normal. The CLT
    guarantees that the *sampling distribution of the mean* converges to N for
    large n, but says nothing about whether X itself is Gaussian. These are
    separate claims and must never be confused.

    Parameters
    ----------
    arr              : 1-D numeric array (NaN-free)
    alpha            : significance level for individual tests
    sw_max_n         : max n for Shapiro-Wilk (omit SW above this)
    normal_threshold : score threshold to classify as 'normal' (default 0.60)

    Returns
    -------
    NormalityResult namedtuple
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n   = len(arr)
    notes: List[str] = []

    if n < 3:
        return NormalityResult(0.0, False, None, np.nan, np.nan,
                               np.nan, np.nan, n, ["n < 3: untestable"])

    score     = 0.0
    sw_p      = None
    ad_stat   = np.nan
    ad_crit5  = np.nan
    skew_val  = float(pd.Series(arr).skew())
    kurt_val  = float(pd.Series(arr).kurt())  # excess kurtosis

    # 1. Shapiro-Wilk (weight 0.35) — omitted above sw_max_n, trivially rejects at large n
    if n <= sw_max_n:
        try:
            _, sw_p = shapiro(arr)
            if sw_p > alpha:
                score += 0.35
                notes.append(f"SW p={sw_p:.4f} ≥ {alpha} (pass)")
            else:
                notes.append(f"SW p={sw_p:.4f} < {alpha} (fail)")
        except Exception as exc:
            notes.append(f"SW failed: {exc}")
    else:
        notes.append(f"SW omitted (n={n} > {sw_max_n})")

    # 2. Anderson-Darling (weight 0.35, or 0.70 when SW is omitted)
    try:
        ad_result = anderson(arr, dist="norm")
        ad_stat   = float(ad_result.statistic)
        crit_idx  = 2  # index for 5% critical value in AD table [15,10,5,2.5,1]%
        ad_crit5  = float(ad_result.critical_values[crit_idx])
        # when SW is skipped, AD absorbs its weight — but AD alone reaching 0.70
        # is not sufficient to declare normality; a moment test must also pass
        ad_weight = 0.35 if n <= sw_max_n else 0.70
        if ad_stat < ad_crit5:
            score += ad_weight
            notes.append(f"AD={ad_stat:.4f} < crit={ad_crit5:.4f} (pass)")
        else:
            notes.append(f"AD={ad_stat:.4f} ≥ crit={ad_crit5:.4f} (fail)")
    except Exception as exc:
        notes.append(f"AD failed: {exc}")

    # 3. Skewness (weight 0.15) — test AND magnitude must both pass (AND, not OR)
    if n >= 8:
        try:
            _, sk_p = skewtest(arr)
            if sk_p > alpha and abs(skew_val) < 0.5:
                score += 0.15
                notes.append(f"skewness={skew_val:+.3f} p={sk_p:.4f} (pass)")
            else:
                notes.append(
                    f"skewness={skew_val:+.3f} p={sk_p:.4f} (fail — "
                    f"{'p<α' if sk_p <= alpha else '|sk|≥0.5'})"
                )
        except Exception:
            if abs(skew_val) < 0.3:
                score += 0.15
                notes.append(f"skewness={skew_val:+.3f} (pass, heuristic |sk|<0.3)")
            else:
                notes.append(f"skewness={skew_val:+.3f} (fail, heuristic)")

    # 4. Kurtosis (weight 0.15) — same AND logic as skewness
    if n >= 20:
        try:
            _, ku_p = kurtosistest(arr)
            if ku_p > alpha and abs(kurt_val) < 1.0:
                score += 0.15
                notes.append(f"kurtosis={kurt_val:+.3f} p={ku_p:.4f} (pass)")
            else:
                notes.append(
                    f"kurtosis={kurt_val:+.3f} p={ku_p:.4f} (fail — "
                    f"{'p<α' if ku_p <= alpha else '|ku|≥1.0'})"
                )
        except Exception:
            if abs(kurt_val) < 0.7:
                score += 0.15
                notes.append(f"kurtosis={kurt_val:+.3f} (pass, heuristic |ku|<0.7)")
            else:
                notes.append(f"kurtosis={kurt_val:+.3f} (fail, heuristic)")

    score = min(score, 1.0)
    # large-n guard: AD alone at 0.70 is too liberal without moment support
    if n > sw_max_n and sw_p is None:
        moment_score = 0.0
        for note in notes:
            if ("skewness" in note or "kurtosis" in note) and "(pass)" in note:
                moment_score += 0.15
        if moment_score == 0.0 and score <= 0.70:
            score = min(score, 0.65)
            notes.append("large-n guard: AD pass without moment support — score capped at 0.65")
    is_normal = score >= normal_threshold

    return NormalityResult(
        score=round(score, 3),
        is_normal=is_normal,
        sw_p=sw_p,
        ad_stat=round(float(ad_stat), 4) if np.isfinite(ad_stat) else np.nan,
        ad_critical_5pct=round(float(ad_crit5), 4) if np.isfinite(ad_crit5) else np.nan,
        skewness=round(float(skew_val), 4),
        kurtosis=round(float(kurt_val), 4),
        n=n,
        notes=notes,
    )


def is_parametric_appropriate(
    arr: np.ndarray,
    alpha: float = 0.05,
    sw_max_n: int = 5_000,
    normal_threshold: float = 0.60,
) -> bool:
    """
    Return True if composite normality evidence supports parametric tests.

    This is NOT a statement that X is Gaussian. It is a practical decision:
    the data distribution is close enough to normal for parametric tests
    to produce reliable p-values under these conditions.

    Callers must still verify variance homogeneity separately when required.
    """
    result = normality_score(arr, alpha=alpha, sw_max_n=sw_max_n,
                             normal_threshold=normal_threshold)
    return result.is_normal


def all_groups_parametric(
    groups: List[np.ndarray],
    alpha: float = 0.05,
    sw_max_n: int = 5_000,
    normal_threshold: float = 0.60,
) -> bool:
    """
    Return True only if ALL groups individually support parametric testing.
    Any group failing the composite normality check makes the whole set non-parametric.
    """
    return all(
        is_parametric_appropriate(g, alpha=alpha, sw_max_n=sw_max_n,
                                  normal_threshold=normal_threshold)
        for g in groups
    )


# =============================================================================
# GROUP TEST SELECTION  (corrected decision tree)
# =============================================================================

def groups_have_equal_variance(
    groups: List[np.ndarray],
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Levene's test for equality of variances.

    This function is ONLY called after confirming that all groups
    support parametric testing. It is meaningless to test variance
    homogeneity before deciding between parametric and non-parametric paths,
    because non-parametric tests (Mann-Whitney, Kruskal) do not assume
    equal variances and the result is irrelevant to their selection.

    Returns
    -------
    (levene_stat, levene_p, equal_variance_holds)
    """
    try:
        stat, p = levene(*groups)
        return float(stat), float(p), bool(p >= alpha)
    except Exception:
        return np.nan, np.nan, True


def choose_group_test(
    groups: List[np.ndarray],
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    homoscedasticity_alpha: float = 0.05,
    normal_threshold: float = 0.60,
) -> Tuple[str, dict]:
    """
    Auto-select the appropriate group comparison test.

    Corrected decision tree
    -----------------------
    Step 1: Assess normality for each group (composite score, not single SW).
            If any group fails → go non-parametric immediately. Levene is
            NOT called in this branch (it is irrelevant to that decision).

    Step 2: If all groups support parametric testing:
            Run Levene's test for variance homogeneity.

            k == 2 groups:
              equal var  → Student t-test
              unequal    → Welch t-test          ← both correct

            k >= 3 groups:
              equal var  → One-way ANOVA          ← classical
              unequal    → Welch ANOVA (Brown-Forsythe approximation)
                           This is the branch that was previously missing.
                           Falling back to Kruskal here is a real power loss.

    Non-parametric path:
            k == 2  → Mann-Whitney U
            k >= 3  → Kruskal-Wallis

    Returns
    -------
    (test_name, diagnostics_dict)

    test_name is one of:
        'student_t', 'welch_t', 'anova', 'welch_anova',
        'mann_whitney', 'kruskal'
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    # Step 1: normality (Levene only on parametric branch)
    normal = all_groups_parametric(
        groups,
        alpha=normality_alpha,
        normal_threshold=normal_threshold,
    )

    diagnostics: dict = {
        "n_groups":        k,
        "all_parametric":  normal,
        "equal_variance":  None,
        "levene_p":        None,
    }

    if not normal:
        test = "mann_whitney" if k == 2 else "kruskal"
        diagnostics["selected_test"] = test
        diagnostics["note"] = (
            "Non-parametric: one or more groups failed composite normality check."
        )
        return test, diagnostics

    lev_stat, lev_p, equal_var = groups_have_equal_variance(
        groups, alpha=homoscedasticity_alpha
    )
    diagnostics["equal_variance"] = equal_var
    diagnostics["levene_p"]       = lev_p

    if k == 2:
        test = "student_t" if equal_var else "welch_t"
    else:
        if equal_var:
            test = "anova"
        else:
            # Welch ANOVA, not Kruskal — data is normal, using non-parametric here loses power
            test = "welch_anova"

    diagnostics["selected_test"] = test
    return test, diagnostics


# =============================================================================
# CORRELATION METHOD SELECTION
# =============================================================================

def choose_correlation_method(
    x: pd.Series,
    y: pd.Series,
    alpha: float = 0.05,
    normal_threshold: float = 0.60,
) -> str:
    """
    Auto-select Pearson vs Spearman based on composite normality of both series.

    Decision rule
    -------------
    Both variables must individually pass the composite normality check for
    Pearson to be appropriate. If either fails → Spearman.

    Spearman measures monotonic association and makes no distributional
    assumption. It is robust to outliers and heavy tails. When in doubt,
    Spearman is the safer choice.

    Note: large n does not justify Pearson — CLT applies to the sample mean,
    not to the data distribution. Log-normal data stays log-normal at any n.

    Returns
    -------
    'pearson' | 'spearman'
    """
    common = x.notna() & y.notna()
    if int(common.sum()) < 3:
        return "spearman"

    x_normal = is_parametric_appropriate(
        x[common].values.astype(float), alpha=alpha, normal_threshold=normal_threshold
    )
    y_normal = is_parametric_appropriate(
        y[common].values.astype(float), alpha=alpha, normal_threshold=normal_threshold
    )
    return "pearson" if (x_normal and y_normal) else "spearman"


# =============================================================================
# EFFECT SIZE FUNCTIONS
# =============================================================================

def compute_cramers_v(chi2_val: float, n: int, k: int) -> float:
    """
    Classic Cramér's V (not bias-corrected). Kept for backward compatibility.

    Prefer compute_cramers_v_bc() when n < 200 or the table has many cells.
    """
    if n <= 0 or k <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, chi2_val) / (n * k)))


def compute_cramers_v_bc(
    chi2_val: float,
    n: int,
    n_rows: int,
    n_cols: int,
) -> float:
    """
    Bias-corrected Cramér's V  (Bergsma & Wicher, 2013).

    The classic formula overestimates V on small samples because chi² has
    positive expected value even under independence. This correction subtracts
    the expected bias before taking the square root, producing an approximately
    unbiased estimator.

    Formula
    -------
    phi²  = max(0, chi²/n - (r-1)(c-1)/(n-1))
    r_tilde = r - (r-1)²/(n-1)
    c_tilde = c - (c-1)²/(n-1)
    V_bc = sqrt(phi² / min(r_tilde-1, c_tilde-1))

    where r = n_rows, c = n_cols.

    Parameters
    ----------
    chi2_val : chi-square statistic
    n        : total observations
    n_rows   : number of rows in the contingency table
    n_cols   : number of columns

    Returns
    -------
    float in [0, 1]  (clipped to 0 if correction produces negative value)
    """
    if n <= 1 or n_rows < 2 or n_cols < 2:
        return 0.0
    phi2     = max(0.0, chi2_val / n)
    phi2_tilde = max(0.0, phi2 - (n_rows - 1) * (n_cols - 1) / (n - 1))
    r_tilde  = n_rows - (n_rows - 1) ** 2 / (n - 1)
    c_tilde  = n_cols - (n_cols - 1) ** 2 / (n - 1)
    denom    = min(r_tilde - 1, c_tilde - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_tilde / denom))


def cramers_v_from_series(x: pd.Series, y: pd.Series, bias_corrected: bool = True) -> float:
    """
    Cramér's V from two categorical Series.

    Parameters
    ----------
    bias_corrected : if True (default) use bias-corrected formula (Bergsma 2013).
                     Set False only for strict backward compatibility.
    """
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n      = int(ct.sum().sum())
    n_rows, n_cols = ct.shape
    if bias_corrected:
        return compute_cramers_v_bc(chi2, n, n_rows, n_cols)
    k = min(n_rows, n_cols) - 1
    return compute_cramers_v(chi2, n, k)


def compute_cohens_d(
    a: np.ndarray,
    b: np.ndarray,
    variant: Literal["pooled", "glass", "hedges"] = "pooled",
) -> float:
    """
    Standardized mean difference — three variants.

    Parameters
    ----------
    a, b    : 1-D numeric arrays (group A and group B)
    variant : which estimator to use

      'pooled' (default — Cohen 1988)
          d = (mean_b - mean_a) / sqrt(((n_a-1)s_a² + (n_b-1)s_b²) / (n_a+n_b-2))
          Assumption: homogeneous variances.  Use when Student t or ANOVA chosen.

      'glass' (Glass 1976)
          Δ = (mean_b - mean_a) / s_a
          Uses only group A (control) SD as denominator. Appropriate when
          treatment changes variances, e.g. one group is a randomized control.
          Use with Welch t-test or when variances are heterogeneous.

      'hedges' (Hedges 1981)
          g = pooled_d × J(df)  where J = 1 - 3/(4*df - 1)
          Small-sample bias correction of pooled Cohen's d.
          Preferred for n < 20 per group.

    Returns
    -------
    float — signed effect size (positive = b > a)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0

    mean_diff = float(b.mean() - a.mean())
    sa = float(a.std(ddof=1))
    sb = float(b.std(ddof=1))

    if variant == "glass":
        denom = sa if sa > 1e-12 else sb
        return mean_diff / max(denom, 1e-12)

    # Pooled SD for 'pooled' and 'hedges'
    pooled_var = ((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2)
    pooled_sd  = float(np.sqrt(max(pooled_var, 1e-12)))
    d = mean_diff / pooled_sd

    if variant == "hedges":
        df = na + nb - 2
        j  = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0 else 1.0
        return d * j

    return d  # 'pooled'


def compute_glass_delta(a: np.ndarray, b: np.ndarray, control: int = 0) -> float:
    """
    Glass Δ — mean difference standardized by the control group SD.

    Parameters
    ----------
    control : index of the control group (0 = a, 1 = b). Default 0.
    """
    return compute_cohens_d(a, b, variant="glass") if control == 0 else compute_cohens_d(b, a, variant="glass")


def compute_hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g — small-sample bias correction of Cohen's d."""
    return compute_cohens_d(a, b, variant="hedges")


def compute_eta_squared(stat: float, n_total: int, k: int, test_type: str = "kruskal") -> float:
    """
    Effect size for group comparison — returns η² (ANOVA) or ε² (Kruskal-Wallis).

    Parameters
    ----------
    stat      : F-statistic (ANOVA) or H-statistic (Kruskal-Wallis)
    n_total   : total observations across all groups
    k         : number of groups
    test_type : 'anova' | 'kruskal'

    Formulas
    --------
    ANOVA   → η² = (F × df_between) / (F × df_between + df_within)
              Eta-squared: proportion of SS_between / SS_total.

    Kruskal → ε² = (H - k + 1) / (n - 1)   [Tomczak & Tomczak 2014]
              Epsilon-squared: rank-based analogue of η².
              NOTE: this is ε² (epsilon-squared), NOT η². The two metrics
              have different denominators and are NOT interchangeable.
              Use compute_epsilon_squared_kruskal() for clarity.

    Note: η² is positively biased in small samples. Use compute_omega_squared()
    for an unbiased estimate, especially when n < 50 per group.
    """
    if test_type == "anova":
        df_between = k - 1
        df_within  = n_total - k
        if df_between <= 0 or df_within <= 0:
            return 0.0
        denom = stat * df_between + df_within
        return float(stat * df_between / denom) if denom > 0 else 0.0
    else:  # kruskal → ε² (Tomczak & Tomczak 2014): denominator is (n-1), NOT (n-k)
        denom = n_total - 1
        return float(max(0.0, (stat - k + 1) / denom)) if denom > 0 else 0.0


def compute_epsilon_squared_kruskal(H: float, n_total: int, k: int) -> float:
    """
    Epsilon-squared ε² for Kruskal-Wallis — the correct rank-based effect size.

    Formula (Tomczak & Tomczak 2014):
        ε² = (H - k + 1) / (n - 1)

    This is the standard non-parametric effect size companion to the KW test.
    It is NOT the same as ANOVA η² — do not mix the two in the same table
    without explicit labelling.

    Thresholds (approximate — context-dependent):
        ε² ≥ 0.14 → large
        ε² ≥ 0.06 → medium
        ε² ≥ 0.01 → small
    """
    denom = n_total - 1
    return float(max(0.0, (H - k + 1) / denom)) if denom > 0 else 0.0


def compute_omega_squared(stat: float, n_total: int, k: int, test_type: str = "anova") -> float:
    """
    Omega-squared ω² — unbiased effect size for group comparison.

    Preferred over η² for small samples (n < 50 per group) because η² is
    positively biased: it overestimates the true proportion of explained
    variance in the population.

    Formula (ANOVA)
    ---------------
    ω² = (SS_between - df_between × MS_within) / (SS_total + MS_within)
       ≈ (F - 1) × (k - 1) / (F × (k - 1) + n - k)       [using F directly]

    Formula (Kruskal)
    -----------------
    ω² ≈ (H - k + 1) / (n - k)  — same as η², correction is negligible
         for rank-based statistics.

    Returns
    -------
    float in [0, 1]  (clipped at 0 — can be negative for very small effects)
    """
    df_between = k - 1
    df_within  = n_total - k
    if df_between <= 0 or df_within <= 0:
        return 0.0

    if test_type == "anova":
        denom = stat * df_between + df_within
        if denom <= 0:
            return 0.0
        return float(max(0.0, (stat - 1) * df_between / denom))
    else:
        # ω² correction is negligible on rank statistics; ε² is the closest analogue
        return compute_epsilon_squared_kruskal(stat, n_total, k)


def compute_rank_biserial(U: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation r_rb from Mann-Whitney U statistic.

    Convention (Kerby 2014)
    -----------------------
    r_rb = 1 - (2U) / (n1 × n2)

    This uses the U statistic returned by scipy.stats.mannwhitneyu(a, b),
    which counts the number of times a value in group a PRECEDES a value
    in group b. The sign of r_rb thus reflects:
        r_rb > 0  → group b tends to have LARGER values
        r_rb < 0  → group b tends to have SMALLER values
        r_rb = 0  → no tendency (stochastic equality)

    The formula is sensitive to which group is passed as a vs b in the
    mannwhitneyu call. Callers must document which group is 'a' and which
    is 'b'. The sign carries directional information — do NOT always abs().

    Parameters
    ----------
    U  : the U statistic for group A (first argument of mannwhitneyu)
    n1 : sample size of group A
    n2 : sample size of group B

    Returns
    -------
    float in [-1, 1]
    """
    denom = n1 * n2
    if denom == 0:
        return 0.0
    return float(1.0 - (2.0 * U) / denom)


# =============================================================================
# CORRELATION HELPERS
# =============================================================================

def corr_with_ci(
    x: pd.Series,
    y: pd.Series,
    method: str = "spearman",
    bootstrap_n: int = 0,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Correlation coefficient with confidence interval.

    CI methods
    ----------
    Pearson  : Fisher z-transform (exact under bivariate normality).
               Valid approximation for most sample sizes with n ≥ 10.

    Spearman : Fisher z-transform applied to Spearman ρ.
               This is an approximation — the z-transform was derived for
               Pearson r and is only asymptotically valid for Spearman.
               For small samples (n < 30) or |ρ| > 0.9, use bootstrap instead.

               Set bootstrap_n > 0 (e.g. 2000) to use BCa bootstrap CI for
               Spearman. This is more accurate at the cost of computation.

    Parameters
    ----------
    x, y         : numeric Series
    method       : 'pearson' | 'spearman'
    bootstrap_n  : if > 0 and method='spearman', use BCa bootstrap CI
                   instead of Fisher z (recommended for n < 30 or |ρ| > 0.9)
    ci_level     : confidence level (default 0.95 → 95% CI)
    random_state : seed for bootstrap reproducibility

    Returns
    -------
    (r, p_value, ci_low, ci_high)
    """
    common = x.notna() & y.notna()
    xv = x[common].values.astype(float)
    yv = y[common].values.astype(float)
    n  = len(xv)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan

    if method == "pearson":
        r, p = pearsonr(xv, yv)
    else:
        r, p = spearmanr(xv, yv)

    r = float(r)
    p = float(p)
    alpha_ci = 1.0 - ci_level
    z_crit = float(_norm.ppf(1.0 - alpha_ci / 2))

    if bootstrap_n > 0 and method == "spearman":
        rng  = np.random.default_rng(random_state)
        boot = []
        for _ in range(bootstrap_n):
            idx = rng.integers(0, n, n)
            try:
                rb, _ = spearmanr(xv[idx], yv[idx])
                boot.append(float(rb))
            except Exception:
                pass
        if len(boot) < 100:
            ci_low = ci_high = r
        else:
            ci_low  = float(np.percentile(boot, alpha_ci / 2 * 100))
            ci_high = float(np.percentile(boot, (1 - alpha_ci / 2) * 100))
    else:
        if abs(r) >= 1.0 or n <= 3:
            ci_low = ci_high = r
        else:
            z      = np.arctanh(r)
            se     = 1.0 / np.sqrt(n - 3)
            ci_low  = float(np.tanh(z - z_crit * se))
            ci_high = float(np.tanh(z + z_crit * se))

    return r, p, ci_low, ci_high


# =============================================================================
# BACKWARD-COMPAT SHIMS  (deprecated — will be removed in a future version)
# =============================================================================

def is_series_normal(
    s: pd.Series,
    alpha: float = 0.05,
    max_n: int = 5_000,
) -> bool:
    """Deprecated. Use is_parametric_appropriate() instead."""
    return is_parametric_appropriate(s.dropna().values.astype(float), alpha=alpha, sw_max_n=max_n)


def is_group_normal(
    arr: np.ndarray,
    alpha: float = 0.05,
    max_n: int = 5_000,
) -> bool:
    """Deprecated. Use is_parametric_appropriate() instead."""
    return is_parametric_appropriate(np.asarray(arr, dtype=float), alpha=alpha, sw_max_n=max_n)


def all_groups_normal(
    groups: List[np.ndarray],
    alpha: float = 0.05,
    max_n: int = 5_000,
) -> bool:
    """Deprecated. Use all_groups_parametric() instead."""
    return all_groups_parametric(groups, alpha=alpha, sw_max_n=max_n)


# =============================================================================
# VIF (VARIANCE INFLATION FACTOR)
# =============================================================================

def compute_vif_series(X_df: pd.DataFrame, col: str) -> float:
    """
    VIF for a single column via OLS R² on all other columns.

    VIF = 1 / (1 - R²)

    Preconditions
    -------------
    - X_df must contain only numeric columns with no NaN (caller's responsibility).
    - col must have non-zero variance; zero-variance columns return VIF=1.0
      because they cannot be linearly predicted by others (R²=0 by definition).
    """
    if X_df[col].std() < 1e-10:
        return 1.0

    feature_cols = [c for c in X_df.columns if c != col]
    if not feature_cols:
        return 1.0

    X_sub = X_df[feature_cols].values
    y_sub = X_df[col].values

    try:
        r2 = LinearRegression().fit(X_sub, y_sub).score(X_sub, y_sub)
        r2 = max(0.0, min(r2, 1.0 - 1e-12))
        return float(1.0 / (1.0 - r2))
    except Exception:
        return np.nan


def compute_vif_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 5.0,
    missing_strategy: Literal["complete_case", "warn"] = "warn",
) -> pd.DataFrame:
    """
    Compute VIF for all specified columns and flag high-VIF features.

    Missing data handling
    ---------------------
    VIF requires a complete numeric matrix. Strategy options:

    'warn' (default)
        Report the number of rows dropped due to missing values before
        computing VIF. If more than 20% of rows are dropped, emit a
        warning because VIF estimates become unreliable on heavily
        reduced samples.

    'complete_case'
        Silent listwise deletion (same as 'warn' but no warning message).

    Parameters
    ----------
    df                : DataFrame
    columns           : columns to evaluate (all numeric if None)
    threshold         : VIF flag threshold (default 5.0)
    missing_strategy  : 'warn' | 'complete_case'

    Returns
    -------
    pd.DataFrame with columns: feature, vif, r2_others, severity, above_threshold
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    num_df = df[columns].select_dtypes(include=[np.number])
    n_before = len(num_df)
    X_num    = num_df.dropna()
    n_after  = len(X_num)
    valid_cols = X_num.columns.tolist()

    if missing_strategy == "warn" and n_before > 0:
        pct_dropped = (n_before - n_after) / n_before * 100
        if pct_dropped > 20:
            logger.warning(
                "VIF: %.1f%% of rows dropped due to missing values (%d → %d). "
                "VIF estimates may be unreliable. Consider imputing before calling VIF.",
                pct_dropped, n_before, n_after,
            )
        elif pct_dropped > 0:
            logger.info("VIF: %d row(s) dropped (%.1f%%) — listwise deletion.", n_before - n_after, pct_dropped)

    if len(valid_cols) < 2:
        return pd.DataFrame(columns=["feature", "vif", "r2_others", "severity", "above_threshold"])

    rows = []
    for col in valid_cols:
        vif = compute_vif_series(X_num, col)
        severity = _vif_severity(vif, threshold)
        r2_others = round(1.0 - 1.0 / vif, 4) if (not np.isnan(vif) and vif > 1.0) else np.nan
        rows.append({
            "feature":         col,
            "vif":             round(vif, 3) if not np.isnan(vif) else np.nan,
            "r2_others":       r2_others,
            "severity":        severity,
            "above_threshold": bool(not np.isnan(vif) and vif > threshold),
        })

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def _vif_severity(vif: float, threshold: float) -> str:
    if np.isnan(vif):
        return "unknown"
    if vif < 1.5:
        return "none"
    if vif < threshold:
        return "moderate"
    if vif < threshold * 2:
        return "high"
    return "severe"


# =============================================================================
# SIGNIFICANCE STARS
# =============================================================================

def significance_stars(p: float) -> str:
    """Return APA-style significance stars for a p-value."""
    if pd.isna(p):
        return "?"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
