
from  typing import List, Dict, Optional
import pandas as pd
from scipy import stats
import  logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder

from ml_framework.visualization.analysis.importance_plots import (
    plot_combined_importance,
    plot_leakage_risk,
)

from  ml_framework.analysis.eda import _cat_cols, _num_cols, _cramers_v
from ml_framework.visualization.base import section_header
logger = logging.getLogger("ml_framework.eda")

# ──────────────────────────────────────────────────────────────────────────────
# 10. LEAKAGE SUSPICION (HEURISTIC — VISUAL)
# ──────────────────────────────────────────────────────────────────────────────

_POST_TREATMENT_KEYWORDS: List[str] = [
    "outcome", "recurrence", "survived", "death", "response",
    "result", "label", "target", "pred", "proba", "flag_target",
]
_DERIVED_PREFIXES: List[str] = [
    "pred_", "proba_", "target_", "leak_", "encoded_target_", "flag_target",
]
_PRE_TREATMENT_ALLOWLIST: List[str] = [
    "age", "sex", "gender", "smoking", "smokingstatus", "bmi",
    "race", "ethnicity", "familyhistory", "history", "region",
    "hospitalregion", "stage",
]


def leakage_exploration(
    df: pd.DataFrame,
    target_col: str,
    domain_allowlist: Optional[List[str]] = None,
    keyword_score: int = 1,
    prefix_score:  int = 3,
    corr_score:    int = 3,
    cramers_score: int = 3,
    proba_score:   int = 2,
    high_threshold:   int = 4,
    medium_threshold: int = 2,
) -> pd.DataFrame:
    """
    Heuristic leakage suspicion scoring — visual risk map output.

    Scoring signals
    ---------------
    +{keyword_score}  post-treatment keyword in column name
    +{prefix_score}   derived-column prefix (pred_, proba_ …)
    +{corr_score}     near-perfect Pearson |r| > 0.95 with numeric target
    +{cramers_score}  near-perfect Cramér's V > 0.95 with categorical target
    +{proba_score}    all values ∈ (0, 1) with high cardinality

    Returns scored DataFrame + leakage risk bar chart.
    This is heuristic suspicion only — not proof of leakage.
    """
    section_header(f"LEAKAGE SUSPICION SCORING — target: {target_col}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    safe_cols = set(_PRE_TREATMENT_ALLOWLIST)
    if domain_allowlist:
        safe_cols.update(c.lower() for c in domain_allowlist)

    target     = df[target_col].dropna()
    is_cat_tgt = not pd.api.types.is_numeric_dtype(target) or target.nunique() <= 15
    rows: List[Dict] = []

    for col in df.columns:
        if col == target_col:
            continue
        if col.lower() in safe_cols:
            continue

        col_lower = col.lower()
        score     = 0
        signals: List[str] = []

        matched_kws = [kw for kw in _POST_TREATMENT_KEYWORDS if kw in col_lower]
        if matched_kws:
            score += keyword_score
            signals.append(f"keyword {matched_kws} (+{keyword_score})")

        if any(col_lower.startswith(p) for p in _DERIVED_PREFIXES):
            score += prefix_score
            signals.append(f"derived prefix (+{prefix_score})")

        if pd.api.types.is_numeric_dtype(df[col]) and not is_cat_tgt:
            try:
                mask = df[col].notna()
                r, _ = stats.pearsonr(df.loc[mask, col], df.loc[mask, target_col])
                if abs(r) > 0.95:
                    score += corr_score
                    signals.append(f"Pearson r={r:+.3f} (+{corr_score})")
            except Exception:
                pass

        if pd.api.types.is_object_dtype(df[col]) and is_cat_tgt:
            try:
                mask = df[col].notna()
                cv   = _cramers_v(df.loc[mask, col], df.loc[mask, target_col])
                if cv > 0.95:
                    score += cramers_score
                    signals.append(f"Cramér's V={cv:.3f} (+{cramers_score})")
            except Exception:
                pass

        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s) > 20 and s.gt(0).all() and s.lt(1).all() and s.nunique() > 20:
                score += proba_score
                signals.append(f"values∈(0,1) (+{proba_score})")

        if score == 0:
            continue

        risk = (
            "HIGH"   if score >= high_threshold   else
            "MEDIUM" if score >= medium_threshold else
            "LOW"
        )
        interp = {
            "HIGH":   "Multiple signals — very likely post-outcome. Exclude before training.",
            "MEDIUM": "One strong signal — verify domain context before deciding.",
            "LOW":    "Weak signal (keyword only) — likely false positive.",
        }[risk]

        rows.append({
            "column": col, "dtype": str(df[col].dtype),
            "score": score, "risk": risk,
            "signals": " | ".join(signals), "interpretation": interp,
        })

    if not rows:
        print("\n No leakage suspicion detected.")
        return pd.DataFrame()

    leak_df = (
        pd.DataFrame(rows).set_index("column")
        .sort_values("score", ascending=False)
    )

    for tier, icon in [("HIGH", "🔴"), ("MEDIUM", "🟡"), ("LOW", "🔵")]:
        subset = leak_df[leak_df["risk"] == tier]
        if subset.empty:
            continue
        print(f"\n  {icon} {tier} ({len(subset)} column(s)):")
        for col_name, row in subset.iterrows():
            print(f"    • {col_name:<30} score={row['score']}  {row['signals']}")
            print(f"      → {row['interpretation']}")

    print(
        "\n   NOTE: Heuristic suspicion only — validate against domain knowledge.\n"
        "          Rule: leakage = information unavailable at prediction time.\n"
    )

    plot_leakage_risk(leak_df)
    return leak_df


# ──────────────────────────────────────────────────────────────────────────────
# 11. FEATURE IMPORTANCE (VISUAL — NO FORMAL TESTS)
# ──────────────────────────────────────────────────────────────────────────────

def feature_importance_exploration(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Pre-modeling feature importance — visual ranking only.

    Metrics used for ranking (visual selection purposes only)
    ---------------------------------------------------------
    association  : Pearson |r| | ANOVA F | Cramér's V | KW η²
    mutual_info  : sklearn MI (model-agnostic)
    rf_importance: Random Forest mean-decrease impurity

    Borda rank aggregation combines all three into one score.
    Results are shown as bar charts — no p-values, no significance stars.
    For formal association tests: statistical_analysis.feature_association_tests()

    Returns
    -------
    pd.DataFrame indexed by feature with columns:
        association, mutual_info, rf_importance,
        assoc_rank, mi_rank, rf_rank, borda_score, metric_type
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")

    section_header(f"FEATURE IMPORTANCE EXPLORATION — target: {target_col}")

    is_cat_tgt = (
        not pd.api.types.is_numeric_dtype(df[target_col])
        or df[target_col].nunique() <= 15
    )

    y = df[target_col].dropna()
    common_idx = y.index

    # ── Prepare X (OHE for categoricals) ─────────────────────────────────────
    num_cols_raw = _num_cols(df, exclude=target_col)
    cat_cols_raw = _cat_cols(df, exclude=target_col)

    X_num = df.loc[common_idx, num_cols_raw].fillna(0) if num_cols_raw else pd.DataFrame(index=common_idx)
    X_cat = pd.DataFrame(index=common_idx)
    ohe   = None

    if cat_cols_raw:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        cat_data  = df.loc[common_idx, cat_cols_raw].fillna("__missing__")
        cat_enc   = ohe.fit_transform(cat_data)
        ohe_names = ohe.get_feature_names_out(cat_cols_raw)
        X_cat     = pd.DataFrame(cat_enc, columns=ohe_names, index=common_idx)

    X = pd.concat([X_num, X_cat], axis=1).fillna(0)
    y = y.loc[common_idx]

    # ── Association metric per feature type ───────────────────────────────────
    assoc: Dict[str, float] = {}
    metric_type: Dict[str, str] = {}

    for col in num_cols_raw:
        try:
            if is_cat_tgt:
                groups = [df.loc[df[target_col] == g, col].dropna().values
                          for g in df[target_col].dropna().unique()]
                if len(groups) >= 2 and all(len(g) >= 5 for g in groups):
                    h, _ = stats.kruskal(*groups)
                    n_tot = sum(len(g) for g in groups)
                    k     = len(groups)
                    eta2  = max(0.0, (h - k + 1) / (n_tot - k))
                    assoc[col] = eta2
                    metric_type[col] = "KW η²"
            else:
                r, _ = stats.pearsonr(df[col].fillna(0), df[target_col].fillna(0))
                assoc[col] = abs(r)
                metric_type[col] = "Pearson |r|"
        except Exception:
            pass

    for col in cat_cols_raw:
        try:
            v = _cramers_v(df[col].dropna(), df[target_col].dropna())
            assoc[col] = v
            metric_type[col] = "Cramér's V"
        except Exception:
            pass

    # ── OHE mapping ───────────────────────────────────────────────────────────
    ohe_col_to_orig: Dict[str, str] = {}
    if cat_cols_raw and ohe is not None:
        for i, orig_col in enumerate(cat_cols_raw):
            for cat_val in ohe.categories_[i][1:]:
                ohe_col_to_orig[f"{orig_col}_{cat_val}"] = orig_col

    def _orig_name(col: str) -> str:
        if col in num_cols_raw:
            return col
        return ohe_col_to_orig.get(col, col)

    # ── Mutual Information ────────────────────────────────────────────────────
    mi_scores: Dict[str, float] = {}
    try:
        mi_fn   = mutual_info_classif if is_cat_tgt else mutual_info_regression
        mi_vals = mi_fn(X, y, random_state=42)
        for col, mi in zip(X.columns, mi_vals):
            orig = _orig_name(col)
            mi_scores[orig] = mi_scores.get(orig, 0.0) + float(mi)
    except Exception as exc:
        logger.warning("Mutual information failed: %s", exc)

    # ── Random Forest importance ──────────────────────────────────────────────
    rf_scores: Dict[str, float] = {}
    try:
        rf_cls = RandomForestClassifier if is_cat_tgt else RandomForestRegressor
        rf     = rf_cls(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        for col, imp in zip(X.columns, rf.feature_importances_):
            orig = _orig_name(col)
            rf_scores[orig] = rf_scores.get(orig, 0.0) + float(imp)
    except Exception as exc:
        logger.warning("Random Forest importance failed: %s", exc)

    # ── Build result DataFrame ────────────────────────────────────────────────
    all_features = list(dict.fromkeys(
        list(assoc.keys()) + list(mi_scores.keys()) + list(rf_scores.keys())
    ))
    rows = []
    for feat in all_features:
        rows.append({
            "feature":       feat,
            "association":   float(assoc.get(feat, 0.0)),
            "mutual_info":   float(mi_scores.get(feat, 0.0)),
            "rf_importance": float(rf_scores.get(feat, 0.0)),
            "metric_type":   metric_type.get(feat, "—"),
        })
    imp_df = pd.DataFrame(rows).set_index("feature")

    for nc in ("association", "mutual_info", "rf_importance"):
        imp_df[nc] = pd.to_numeric(imp_df[nc], errors="coerce").fillna(0.0)

    def _rank_norm(s: pd.Series) -> pd.Series:
        s   = pd.to_numeric(s, errors="coerce").fillna(0.0)
        r   = s.rank(method="average", ascending=True)
        rng = r.max() - r.min()
        return ((r - r.min()) / rng).round(4) if rng > 0 else pd.Series(0.5, index=s.index)

    imp_df["assoc_rank"] = _rank_norm(imp_df["association"])
    imp_df["mi_rank"]    = _rank_norm(imp_df["mutual_info"])
    imp_df["rf_rank"]    = _rank_norm(imp_df["rf_importance"])
    imp_df["borda_score"] = imp_df[["assoc_rank", "mi_rank", "rf_rank"]].mean(axis=1).round(4)
    imp_df = imp_df.sort_values("borda_score", ascending=False).head(top_n)

    print(f"\n  Top {top_n} features — Borda rank (association | MI | RF) — visual ranking only:\n")
    num_display = ["association", "mutual_info", "rf_importance", "borda_score"]
    print(pd.concat([imp_df[num_display].round(3), imp_df[["metric_type"]]], axis=1).to_string())
    print("\n  → For formal tests: statistical_analysis.feature_association_tests()")

    plot_combined_importance(imp_df, target_col)
    return imp_df

