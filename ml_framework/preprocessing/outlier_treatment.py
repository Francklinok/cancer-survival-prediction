"""
outlier_treatment.py — Object-oriented outlier treatment system.

Provides OutlierTreatmentSystem: a production-ready class that applies
various treatment strategies to detected outliers:
  - cap        : IQR-based clipping
  - winsorize  : scipy winsorization
  - transform  : Yeo-Johnson / log1p
  - impute     : KNN imputation
  - scale      : RobustScaler
  - remove     : row deletion

Usage:
    from ml_framework.preprocessing.outlier_treatment import OutlierTreatmentSystem

    config = OutlierTreatmentConfig(method='winsorize')
    system = OutlierTreatmentSystem(config)
    df_clean = system.apply(df, outliers_dict)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler

logger = logging.getLogger("ml_framework.outlier_treatment")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────


class OutlierTreatmentConfig:
    """
    Configuration for the outlier treatment system.

    Attributes
    ----------
    method           : strategy — 'cap' | 'winsorize' | 'transform' | 'impute' | 'scale' | 'remove'
    iqr_multiplier   : IQR multiplier for the 'cap' method (default 1.5)
    winsorize_limits : (lower_pct, upper_pct) for winsorize (default (0.05, 0.05))
    n_neighbors      : KNN neighbors for 'impute' (default 5)
    """

    def __init__(
        self,
        method: str = "winsorize",
        iqr_multiplier: float = 1.5,
        winsorize_limits: tuple = (0.05, 0.05),
        n_neighbors: int = 5,
    ):
        self.method           = method
        self.iqr_multiplier   = iqr_multiplier
        self.winsorize_limits = winsorize_limits
        self.n_neighbors      = n_neighbors

    def validate(self) -> None:
        valid_methods = {"cap", "winsorize", "transform", "impute", "scale", "remove"}
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method: '{self.method}'. Valid options: {valid_methods}")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be > 0.")
        if not (0 <= self.winsorize_limits[0] < 0.5 and 0 <= self.winsorize_limits[1] < 0.5):
            raise ValueError("winsorize_limits: each limit must be in [0, 0.5).")


# ──────────────────────────────────────────────────────────────────────────────
# TREATMENT SYSTEM
# ──────────────────────────────────────────────────────────────────────────────


class OutlierTreatmentSystem:
    """
    Outlier treatment system.

    Consistently applies a treatment strategy to a DataFrame using the
    outliers dictionary produced by identify_outliers().

    Parameters
    ----------
    config : OutlierTreatmentConfig — strategy configuration

    Methods
    -------
    apply(data, outliers) → pd.DataFrame
    summary(original, treated) → pd.DataFrame
    """

    def __init__(self, config: Optional[OutlierTreatmentConfig] = None):
        if config is None:
            config = OutlierTreatmentConfig()
        self.config = config
        self.config.validate()

    def apply(
        self,
        data: pd.DataFrame,
        outliers: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply the treatment strategy to detected outliers.

        Parameters
        ----------
        data     : original DataFrame
        outliers : dict {col_name: {'indices': [...], 'count': int, ...}}
                   produced by identify_outliers()

        Returns
        -------
        pd.DataFrame — treated DataFrame
        """
        if data.empty or not outliers:
            return data.copy()

        result = data.copy()
        n_treated = 0

        for col, info in outliers.items():
            if col not in result.columns:
                logger.debug("Column '%s' absent — skipped.", col)
                continue

            idx = info.get("indices", [])
            if len(idx) == 0:
                continue

            series = result[col].copy()

            try:
                if self.config.method == "cap":
                    series.loc[idx] = self._cap(series, idx)

                elif self.config.method == "winsorize":
                    series.loc[idx] = self._winsorize(series, idx)

                elif self.config.method == "transform":
                    series.loc[idx] = self._transform(series, idx)

                elif self.config.method == "impute":
                    series = self._impute(series, idx)

                elif self.config.method == "scale":
                    series = self._scale(series)

                elif self.config.method == "remove":
                    result = result.drop(index=[i for i in idx if i in result.index])
                    n_treated += len(idx)
                    continue

                result[col] = series
                n_treated += 1

            except Exception as exc:
                logger.error(
                    "Treatment '%s' on column '%s': %s",
                    self.config.method, col, exc,
                )

        logger.info(
            "OutlierTreatmentSystem(%s): %d columns treated.",
            self.config.method, n_treated,
        )
        return result

    # ── Private treatment methods ─────────────────────────────────────────────

    def _cap(self, series: pd.Series, idx: list) -> pd.Series:
        """IQR-based clipping (clip to IQR bounds)."""
        clean = series.dropna()
        if clean.empty:
            return series.loc[idx]

        q1  = float(clean.quantile(0.25))
        q3  = float(clean.quantile(0.75))
        iqr = q3 - q1

        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr

        return series.loc[idx].clip(lower, upper)

    def _winsorize(self, series: pd.Series, idx: list) -> pd.Series:
        """Winsorization (replace with boundary percentile values)."""
        clean = series.dropna()
        if clean.empty:
            return series.loc[idx]

        lo_pct, hi_pct = self.config.winsorize_limits
        lower = float(clean.quantile(lo_pct))
        upper = float(clean.quantile(1 - hi_pct))

        return series.loc[idx].clip(lower, upper)

    def _transform(self, series: pd.Series, idx: list) -> pd.Series:
        """Yeo-Johnson transformation (fallback to log1p if it fails)."""
        clean = series.dropna()
        if clean.empty:
            return series.loc[idx]

        try:
            transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            arr = clean.values.reshape(-1, 1)
            transformed = transformer.fit_transform(arr).flatten()
            s_transformed = pd.Series(transformed, index=clean.index)
            return s_transformed.reindex(series.index).loc[idx]
        except Exception:
            shifted = clean - clean.min() + 1 if clean.min() <= 0 else clean
            return np.log1p(shifted).reindex(series.index).loc[idx]

    def _impute(self, series: pd.Series, outlier_idx: list) -> pd.Series:
        """KNN imputation (outliers → NaN, then impute)."""
        temp = series.copy()
        valid_idx = [i for i in outlier_idx if i in temp.index]
        temp.loc[valid_idx] = np.nan

        if temp.dropna().empty:
            return series

        if len(temp.dropna()) < self.config.n_neighbors:
            fill_value = float(temp.median())
            series.loc[valid_idx] = fill_value
            return series

        imputer = KNNImputer(n_neighbors=self.config.n_neighbors)
        arr     = temp.values.reshape(-1, 1)
        imputed = imputer.fit_transform(arr).flatten()

        result = series.copy()
        result.iloc[:len(imputed)] = imputed
        return result

    def _scale(self, series: pd.Series) -> pd.Series:
        """Robust scaling (median / IQR) across the entire series."""
        clean = series.dropna()
        if clean.empty:
            return series

        scaler = RobustScaler()
        arr    = clean.values.reshape(-1, 1)
        scaled = scaler.fit_transform(arr).flatten()

        result = series.copy()
        result.loc[clean.index] = scaled
        return result

    # ── Summary report ────────────────────────────────────────────────────────

    def summary(self, original: pd.DataFrame, treated: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame comparing descriptive statistics before/after treatment.

        Parameters
        ----------
        original : DataFrame before treatment
        treated  : DataFrame after treatment

        Returns
        -------
        pd.DataFrame with columns: col, mean_before, mean_after, std_before, std_after, delta_std
        """
        rows = []
        common_cols = [
            c for c in original.columns
            if c in treated.columns
            and original[c].dtype in [np.float64, np.int64, float, int]
        ]

        for col in common_cols:
            before = original[col].dropna()
            after  = treated[col].dropna()
            rows.append({
                "col":         col,
                "mean_before": round(float(before.mean()), 4),
                "mean_after":  round(float(after.mean()),  4),
                "std_before":  round(float(before.std()),  4),
                "std_after":   round(float(after.std()),   4),
                "delta_std":   round(float(after.std()) - float(before.std()), 4),
            })

        return pd.DataFrame(rows)
