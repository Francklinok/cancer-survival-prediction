"""
feature_engineering.py — Adaptive Feature Engine.


Architecture
------------
AdaptiveFeatureEngine          — main class, learns which features to keep
  .fit(X, y)                   — evaluates every candidate against CV baseline
  .transform(X)                — applies only the retained transformations
  .fit_transform(X, y)         — fit + transform in one call

engineer_features(df, target)  — functional wrapper (backward-compatible API)

Anti-noise mechanisms
---------------------
1. Candidate generation is bounded:
     - interactions: mutual-information top-k pairs only (not all combinations)
     - polynomials: only for features that individually correlate with target
     - binning: only for non-uniform (skewed) distributions

2. Each candidate is evaluated via a fast CV probe (3-fold, LightGBM or RF):
     - retained only if delta_CV >= min_gain (default 0.002)
     - evaluation done on X_train signal, never on held-out data

3. Structural leakage guards:
     - SurvivalMonths / temporal columns → flagged, skipped by default
     - Trivial interactions (|r| > 0.95 with either parent) → dropped
     - Constant / near-constant features (std < 1e-6) → dropped

4. Budget cap: at most max_features new columns are retained.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger("ml_framework.feature_engineering")

_LEAKAGE_COLS: Set[str] = {
    "SurvivalMonths", "Survival_Months", "survival_months",
    "PatientID", "Patient_ID", "ID",
}

_PROBE_SAMPLE = 3_000


# =============================================================================
# FAST CV PROBE
# =============================================================================


def _probe_score(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    scoring: str = "f1_macro",
) -> float:
    """
    Fast 3-fold CV score on a small probe dataset.
    Uses LightGBM if available, otherwise RandomForest (n_estimators=30).
    """
    X_arr = X.fillna(0).values
    y_arr = y.values

    if len(X_arr) > _PROBE_SAMPLE:
        idx = np.random.choice(len(X_arr), _PROBE_SAMPLE, replace=False)
        X_arr, y_arr = X_arr[idx], y_arr[idx]

    try:
        from lightgbm import LGBMClassifier
        probe = LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            n_jobs=1, verbose=-1, random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        probe = RandomForestClassifier(
            n_estimators=30, max_depth=5, n_jobs=1, random_state=42,
        )

    try:
        scores = cross_val_score(probe, X_arr, y_arr, cv=cv,
                                 scoring=scoring, n_jobs=1)
        return float(scores.mean())
    except Exception:
        return 0.0


# =============================================================================
# ADAPTIVE FEATURE ENGINE
# =============================================================================


class AdaptiveFeatureEngine(BaseEstimator, TransformerMixin):
    """
    Learns which engineered features genuinely improve CV signal.

    Parameters
    ----------
    target_col          : name of the target column (excluded from transforms)
    max_features        : hard cap on new features retained (default 20)
    min_gain            : minimum CV F1-macro delta to keep a feature (default 0.002)
    max_interaction_pairs: max candidate interaction pairs evaluated (default 30)
    cv_folds            : number of CV folds for the probe (default 3)
    scoring             : sklearn scoring string (default 'f1_macro')
    leakage_cols        : extra column names to skip (merged with _LEAKAGE_COLS)
    skip_leakage_cols   : if True, skip known temporal/leakage columns
    domain_features_fn  : optional callable(df, target_col) -> List[str] that
                          adds domain-specific engineered features in place
                          (e.g. ml_framework.domain.medical.add_oncology_features).
                          These are applied unconditionally (not CV-gain-filtered),
                          since they encode known domain relationships rather
                          than candidate noise. None (default) = no domain
                          features — framework-neutral behaviour.
    add_medical         : deprecated back-compat alias — True is equivalent to
                          domain_features_fn=add_oncology_features. Prefer
                          passing domain_features_fn explicitly.
    verbose             : print a selection table
    random_state        : reproducibility seed
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        max_features: int = 20,
        min_gain: float = 0.002,
        max_interaction_pairs: int = 30,
        cv_folds: int = 3,
        scoring: str = "f1_macro",
        leakage_cols: Optional[List[str]] = None,
        skip_leakage_cols: bool = True,
        domain_features_fn=None,
        add_medical: bool = False,
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.target_col           = target_col
        self.max_features         = max_features
        self.min_gain             = min_gain
        self.max_interaction_pairs = max_interaction_pairs
        self.cv_folds             = cv_folds
        self.scoring              = scoring
        self.leakage_cols         = set(leakage_cols or []) | _LEAKAGE_COLS
        self.skip_leakage_cols    = skip_leakage_cols
        if domain_features_fn is None and add_medical:
            from ml_framework.domain.medical.oncology_features import add_oncology_features
            domain_features_fn = add_oncology_features
        self.domain_features_fn  = domain_features_fn
        self.add_medical          = add_medical
        self.verbose              = verbose
        self.random_state         = random_state

        # Learned state (set during fit)
        self.retained_: List[Dict] = []      # list of {name, type, delta_cv, formula}
        self.baseline_cv_: float   = 0.0
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveFeatureEngine":
        """
        Evaluate all candidate features against a CV baseline.
        Retain only those that improve F1-macro by at least min_gain.
        """
        np.random.seed(self.random_state)

        self.feature_names_in_ = X.columns.tolist()
        target = self.target_col

        num_cols = self._num_cols(X)
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        self.baseline_cv_ = _probe_score(X[num_cols], y, cv, self.scoring)
        if self.verbose:
            print(f"\n  AdaptiveFeatureEngine — baseline CV {self.scoring}: "
                  f"{self.baseline_cv_:.4f}")
            print(f"  min_gain={self.min_gain}  max_features={self.max_features}  "
                  f"max_pairs={self.max_interaction_pairs}")
            print(f"  {'Candidate':<45} {'type':<14} {'delta CV':>9}  {'keep'}")
            print("  " + "-" * 80)

        candidates = self._generate_candidates(X, y, num_cols)

        self.retained_ = []
        budget = self.max_features

        for cand in candidates:
            if budget <= 0:
                break

            name     = cand["name"]
            cand_val = cand["values"]

            if cand_val.std() < 1e-6:
                self._vprint(name, cand["type"], None, False, "constant")
                continue

            # skip if nearly identical to a parent — no new information
            if "parents" in cand:
                max_parent_r = max(
                    abs(X[p].corr(cand_val)) for p in cand["parents"] if p in X.columns
                )
                if max_parent_r > 0.95:
                    self._vprint(name, cand["type"], None, False,
                                 f"|r|={max_parent_r:.2f} trivial")
                    continue

            X_aug = pd.concat([X[num_cols].fillna(0),
                               cand_val.rename(name).fillna(0)], axis=1)
            score  = _probe_score(X_aug, y, cv, self.scoring)
            delta  = score - self.baseline_cv_
            keep   = delta >= self.min_gain

            self._vprint(name, cand["type"], delta, keep)

            if keep:
                self.retained_.append({
                    "name":     name,
                    "type":     cand["type"],
                    "delta_cv": round(delta, 5),
                    "formula":  cand.get("formula", ""),
                    "parents":  cand.get("parents", []),
                })
                budget -= 1

        if self.verbose:
            print(f"\n  Retained {len(self.retained_)}/{len(candidates)} candidates  "
                  f"(budget cap: {self.max_features})")

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply only the retained transformations to X.

        Returns
        -------
        (df_out, new_feature_names)
        """
        df_out    = X.copy()
        added: List[str] = []

        if self.domain_features_fn is not None:
            domain_new = self.domain_features_fn(df_out, self.target_col)
            added.extend(domain_new)

        for rec in self.retained_:
            name    = rec["name"]
            formula = rec["formula"]
            ftype   = rec["type"]

            if name in df_out.columns:
                continue

            try:
                val = self._apply_formula(df_out, formula, ftype, rec)
                if val is not None:
                    df_out[name] = val.fillna(0)
                    added.append(name)
            except Exception as exc:
                logger.warning("transform: failed to create '%s': %s", name, exc)

        return df_out, added

    def fit_transform(                             
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        return self.fit(X, y).transform(X)

    def _generate_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cols: List[str],
    ) -> List[Dict]:
        """
        Build candidate list, ordered by estimated informativeness.
        Uses mutual information (fast, non-linear) to rank parent features.
        """
        candidates: List[Dict] = []

        try:
            mi_scores = mutual_info_classif(
                X[num_cols].fillna(0), y,
                discrete_features=False, random_state=self.random_state,
            )
            mi_ranked = sorted(
                zip(num_cols, mi_scores), key=lambda t: t[1], reverse=True
            )
        except Exception:
            mi_ranked = [(c, 1.0) for c in num_cols]

        top_mi_cols = [c for c, _ in mi_ranked[:min(12, len(mi_ranked))]]

        # 1. log transform — only when all values are positive and distribution is skewed
        for col in num_cols:
            if col in self.leakage_cols and self.skip_leakage_cols:
                continue
            series = X[col].fillna(0)
            if (series > 0).all() and abs(float(series.skew())) > 1.0:
                name = f"log_{col}"
                candidates.append({
                    "name":    name,
                    "type":    "log",
                    "values":  np.log1p(series),
                    "formula": f"log1p({col})",
                    "parents": [col],
                })

        # 2. pairwise interactions limited to high-MI features to avoid combinatorial explosion
        pairs = list(combinations(top_mi_cols, 2))
        mi_dict = dict(mi_ranked)
        pairs.sort(key=lambda p: mi_dict.get(p[0], 0) + mi_dict.get(p[1], 0),
                   reverse=True)
        pairs = pairs[: self.max_interaction_pairs]

        for c1, c2 in pairs:
            if c1 in self.leakage_cols or c2 in self.leakage_cols:
                continue

            prod_val = X[c1] * X[c2]
            candidates.append({
                "name":    f"inter_{c1}_x_{c2}",
                "type":    "product",
                "values":  prod_val,
                "formula": f"{c1}*{c2}",
                "parents": [c1, c2],
            })

            # ratio skipped when more than 15% of denominator values are zero
            denom = X[c2].replace(0, np.nan)
            if denom.notna().mean() > 0.85:
                ratio_val = (X[c1] / denom).fillna(0).clip(-1e6, 1e6)
                candidates.append({
                    "name":    f"ratio_{c1}_over_{c2}",
                    "type":    "ratio",
                    "values":  ratio_val,
                    "formula": f"{c1}/{c2}",
                    "parents": [c1, c2],
                })

        # 3. squared terms for the 6 most informative features
        for col, mi in mi_ranked[:6]:
            if col in self.leakage_cols:
                continue
            sq_val = X[col] ** 2
            candidates.append({
                "name":    f"{col}_sq",
                "type":    "polynomial",
                "values":  sq_val,
                "formula": f"{col}**2",
                "parents": [col],
            })

        # 4. quantile bins — skipped for near-uniform distributions (IQR/std < 1.2)
        for col in num_cols[:8]:
            if col in self.leakage_cols:
                continue
            series = X[col].dropna()
            iqr = float(series.quantile(0.75) - series.quantile(0.25))
            std = float(series.std())
            if std > 1e-6 and iqr / std < 1.2:
                continue
            try:
                bin_val = pd.qcut(X[col], q=5, labels=False, duplicates="drop")
                candidates.append({
                    "name":    f"bin_{col}",
                    "type":    "binning",
                    "values":  bin_val.fillna(-1),
                    "formula": f"qcut5({col})",
                    "parents": [col],
                })
            except Exception:
                pass

        return candidates

    def _num_cols(self, X: pd.DataFrame) -> List[str]:
        skip = {self.target_col} | (self.leakage_cols if self.skip_leakage_cols else set())
        return [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in skip and X[c].nunique() > 2
        ]

    def _apply_formula(
        self,
        df: pd.DataFrame,
        formula: str,
        ftype: str,
        rec: Dict,
    ) -> Optional[pd.Series]:
        parents = rec.get("parents", [])
        missing = [p for p in parents if p not in df.columns]
        if missing:
            logger.warning("transform: parent column(s) %s not found — skipping %s",
                           missing, rec["name"])
            return None

        if ftype == "log":
            col = parents[0]
            s = df[col].fillna(0)
            return np.log1p(s.clip(lower=0))

        if ftype == "product":
            return df[parents[0]] * df[parents[1]]

        if ftype == "ratio":
            denom = df[parents[1]].replace(0, np.nan)
            return (df[parents[0]] / denom).fillna(0).clip(-1e6, 1e6)

        if ftype == "polynomial":
            return df[parents[0]] ** 2

        if ftype == "binning":
            try:
                return pd.qcut(df[parents[0]], q=5, labels=False, duplicates="drop").fillna(-1)
            except Exception:
                return None

        return None

    def _vprint(
        self,
        name: str,
        ftype: str,
        delta: Optional[float],
        keep: bool,
        note: str = "",
    ) -> None:
        if not self.verbose:
            return
        delta_s = f"{delta:+.4f}" if delta is not None else "  n/a "
        status  = "YES" if keep else f"no  {note}"
        print(f"  {name:<45} {ftype:<14} {delta_s:>9}  {status}")

    def get_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising retained features."""
        if not self.retained_:
            return pd.DataFrame(columns=["name", "type", "delta_cv", "formula"])
        return (
            pd.DataFrame(self.retained_)[["name", "type", "delta_cv", "formula"]]
            .sort_values("delta_cv", ascending=False)
            .reset_index(drop=True)
        )


# =============================================================================
# DOMAIN-SPECIFIC FEATURES — backward-compatible re-export
# =============================================================================
#
# The oncology-specific implementation that used to live here moved to
# ml_framework.domain.medical.oncology_features.add_oncology_features()
# (identical formulas and behaviour) so this module stays domain-neutral.
# _add_medical_features is kept as a deprecated alias for any external code
# still importing it directly.


def _add_medical_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> List[str]:
    """Deprecated alias — see ml_framework.domain.medical.add_oncology_features."""
    from ml_framework.domain.medical.oncology_features import add_oncology_features
    return add_oncology_features(df, target_col)


# =============================================================================
# FUNCTIONAL API
# =============================================================================


def engineer_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    max_features: int = 20,
    min_gain: float = 0.002,
    max_interaction_pairs: int = 30,
    cv_folds: int = 3,
    skip_leakage_cols: bool = True,
    domain_features_fn=None,
    add_medical: bool = False,
    verbose: bool = True,
    random_state: int = 42,
    # legacy parameters — accepted but ignored
    # add_interactions: bool = True,
    # add_polynomial: bool = False,
    # poly_degree: int = 2,
    # add_binning: bool = True,
    # n_bins: int = 5,
    # add_log_transform: bool = True,
    # add_medical_features: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adaptive feature engineering — functional wrapper.

    Calls AdaptiveFeatureEngine.fit_transform() when a target is provided.
    Falls back to domain features only when target is absent (unsupervised path).

    Parameters
    ----------
    df                    : input DataFrame (post-imputation, pre-normalization)
    target_col            : target column name (required for CV evaluation)
    max_features          : hard cap on retained new features
    min_gain              : minimum CV F1-macro delta to retain a feature
    max_interaction_pairs : max candidate pairs evaluated
    cv_folds              : CV folds for probe
    skip_leakage_cols     : skip SurvivalMonths and similar temporal columns
    domain_features_fn    : optional callable(df, target_col) -> List[str] adding
                             domain-specific features unconditionally (see
                             AdaptiveFeatureEngine docstring). None (default) =
                             framework-neutral, no domain features added.
    add_medical           : deprecated back-compat alias for
                             domain_features_fn=ml_framework.domain.medical.add_oncology_features
    verbose               : print selection table
    random_state          : reproducibility seed

    Returns
    -------
    df_engineered : pd.DataFrame
    new_features  : List[str]
    """
    if domain_features_fn is None and add_medical:
        from ml_framework.domain.medical.oncology_features import add_oncology_features
        domain_features_fn = add_oncology_features

    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        engine = AdaptiveFeatureEngine(
            target_col=target_col,
            max_features=max_features,
            min_gain=min_gain,
            max_interaction_pairs=max_interaction_pairs,
            cv_folds=cv_folds,
            skip_leakage_cols=skip_leakage_cols,
            domain_features_fn=domain_features_fn,
            verbose=verbose,
            random_state=random_state,
        )
        df_eng, new_feats = engine.fit_transform(X, y)
        df_eng[target_col] = y.values

        if verbose and engine.retained_:
            print("\n  Retained features summary:")
            print(engine.get_summary().to_string(index=False))

    else:
        df_eng  = df.copy()
        new_feats = domain_features_fn(df_eng, target_col) if domain_features_fn else []
        if verbose:
            print(f"\n  Feature Engineering (no target): "
                  f"{len(new_feats)} domain feature(s) added.")

    return df_eng, new_feats
