"""
config.py — Central configuration for the medical ml_framework.

All constants, dataclasses, and global parameters are consolidated here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL PANDAS / MATPLOTLIB / NUMPY CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1200)
pd.set_option("display.float_format", "{:.4f}".format)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

import sys as _sys

def _setup_logging() -> None:
    root = logging.getLogger()
    # Remove any existing handlers to avoid duplication on re-import
    for _h in root.handlers[:]:
        root.removeHandler(_h)
    _stream = _sys.stdout
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass
    _handler = logging.StreamHandler(_stream)
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.setLevel(logging.INFO)
    root.addHandler(_handler)

_setup_logging()
logger = logging.getLogger("ml_framework")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class NormalizationConfig:
    """Parameters for the automatic normalization strategy."""

    alpha: float = 0.05
    """Significance threshold for normality tests."""

    skew_threshold: float = 1.0
    """Moderate skewness → log/sqrt transform."""

    high_skew_threshold: float = 2.0
    """High skewness → Box-Cox / Yeo-Johnson."""

    kurtosis_threshold: float = 3.0
    """Excess kurtosis → heavy-tail detection."""

    normal_skew_tol: float = 0.5
    """Symmetry tolerance to consider a distribution as normal."""

    normal_kurt_tol: float = 1.0
    """Kurtosis tolerance to consider a distribution as normal."""

    outlier_threshold: float = 10.0
    """Percentage of IQR outliers above which winsorization/robust scaler is applied (e.g. 10.0 = 10%)."""

    cv_high: float = 1.0
    """High coefficient of variation → robust normalization."""

    cv_very_high: float = 2.0
    """Very high coefficient of variation → specialized scaling."""

    range_threshold: float = 1_000.0
    """Large range → min-max standardization."""

    winsorize_lower: float = 0.05
    """Lower bound for winsorization."""

    winsorize_upper: float = 0.95
    """Upper bound for winsorization."""


@dataclass
class ModelConfig:
    """General parameters for the modeling pipeline."""

    target_column: str = "target"
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    scoring: str = "roc_auc"
    models_to_test: List[str] = field(
        default_factory=lambda: ["rf", "gb", "lr", "svm"]
    )
    perform_hyperparameter_tuning: bool = True
    n_iter_bayesian: int = 50
    n_jobs: int = -1
    early_stopping_rounds: Optional[int] = 10
    threshold: float = 0.50


@dataclass
class DataConfig:
    """Parameters for data loading and validation."""

    supported_formats: List[str] = field(
        default_factory=lambda: [".csv", ".xlsx", ".xls", ".parquet", ".json"]
    )
    encoding: str = "utf-8"
    separator: str = ","
    missing_strategy: str = "mice"          # 'mice' | 'knn' | 'miss_forest' | 'simple'
    outlier_method: str = "iqr"             # 'iqr' | 'zscore' | 'modified_zscore' | 'IsolationForest'
    contamination: float = 0.05
    max_missing_ratio: float = 0.50         # columns with > 50% NaN → dropped
    min_variance: float = 1e-8              # minimum variance to retain a column


@dataclass
class ReportConfig:
    """Parameters for report generation."""

    output_dir: str = "reports"
    save_plots: bool = True
    dpi: int = 150
    figure_format: str = "png"
    risk_thresholds: dict = field(
        default_factory=lambda: {"Low": 0.30, "Medium": 0.60, "High": 0.80}
    )
    top_n_features: int = 15


@dataclass
class FrameworkConfig:
    """Master configuration aggregating all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Known ordinal columns for the medical domain (cancer dataset)
    ordinal_columns: List[str] = field(
        default_factory=lambda: [
            "Cancer_Stage", "Obesity_BMI", "Physical_Activity",
            "Diet_Risk", "Economic_Classification", "Healthcare_Access",
            "SmokingStatus", "Stage", "TreatmentResponse",
            "Survival_Category", "Age_Group", "BMI_Category",
            "Tumor_Size_Category",
        ]
    )

    sensitive_attributes: List[str] = field(
        default_factory=lambda: ["Gender", "Race/Ethnicity", "Age_Group"]
    )


# Default instance accessible throughout the framework
DEFAULT_CONFIG = FrameworkConfig()
