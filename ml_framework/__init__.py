"""
ml_framework — General-purpose, ML framework.

Domain-neutral by default;
An optional medical/oncology domain pack
(ml_framework.domain.medical) is available for the reference dataset .

Quick import:

    from ml_framework import generate_model_card, diagnostic_analysis
    from ml_framework.orchestration.pipeline import MLPipeline
    from ml_framework.config.config import FrameworkConfig, ModelConfig

    pipeline = MLPipeline()
    pipeline.run("data.csv", target_column="target")

    # Medical/oncology reference dataset:
    #   pipeline = MLPipeline(FrameworkConfig.from_domain("medical"))
    #   pipeline.run("data.csv", target_column="Recurrence")
"""

__version__ = "2.0.0"
__author__ = "ml_framework"

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
from ml_framework.config.config import (
    DEFAULT_CONFIG,
    DataConfig,
    FrameworkConfig,
    ModelConfig,
    NormalizationConfig,
    ReportConfig,
)

# ── Main pipeline ─────────────────────────────────────────────────────────────
from ml_framework.orchestration.pipeline import MLPipeline, MedicalMLPipeline

# ── Data loading ─────────────────────────────────────────────────────────────
from ml_framework.services.data_loading import load_data

# ── Analysis & diagnostics ──────────────────────────────────────────────────
from ml_framework.analysis.diagnostic_analysis import (
    ci_str,
    diagnostic_analysis,
    causal_analysis,
)
from ml_framework.visualization.analysis.visualization_system import VisualizationSystem
from ml_framework.analysis.data_drift import bucket

# ── Preprocessing ─────────────────────────────────────────────────────────────
from ml_framework.preprocessing.outlier_treatment import (
    OutlierTreatmentConfig,
    OutlierTreatmentSystem,
)
from ml_framework.preprocessing.missing_value import (
    missing_data_handling,
    replace_value,
)

# ── Model card ───────────────────────────────────────────────────────────────
from ml_framework.modeling.model_card import (
    generate_model_card,
    save_model_card,
    print_model_card,
)

__all__ = [
    # Config
    "MLPipeline",
    "MedicalMLPipeline",
    "FrameworkConfig",
    "ModelConfig",
    "DataConfig",
    "NormalizationConfig",
    "ReportConfig",
    "DEFAULT_CONFIG",
    # Data loading
    "load_data",
    # Analysis
    "ci_str",
    "diagnostic_analysis",
    "causal_analysis",
    "VisualizationSystem",
    "bucket",
    # Preprocessing
    "OutlierTreatmentConfig",
    "OutlierTreatmentSystem",
    "missing_data_handling",
    "replace_value",
    # Model card
    "generate_model_card",
    "save_model_card",
    "print_model_card",
]
