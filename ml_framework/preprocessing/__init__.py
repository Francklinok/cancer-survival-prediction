# ml_framework/preprocessing package

from ml_framework.preprocessing.clean import clean_dataframe
from ml_framework.preprocessing.encoding import (
    encode_dataframe,
    encodage_processing,  # backward-compat alias
    target_encoding,
    frequency_encoding,
    one_hot_encode,
)
from ml_framework.preprocessing.outlier_treatment import (
    OutlierTreatmentConfig,
    OutlierTreatmentSystem,
)
from ml_framework.preprocessing.missing_value import (
    missing_data_handling,
    replace_value,
)
from ml_framework.preprocessing.apply_normalisation import (
    apply_normalization,
    _apply_transform,
    _apply_scaler,
)

__all__ = [
    # Cleaning
    "clean_dataframe",
    # Encoding
    "encode_dataframe",
    "encodage_processing",
    "target_encoding",
    "frequency_encoding",
    "one_hot_encode",
    # Outlier treatment
    "OutlierTreatmentConfig",
    "OutlierTreatmentSystem",
    # Missing values
    "missing_data_handling",
    "replace_value",
    # Normalization application
    "apply_normalization",
    "_apply_transform",
    "_apply_scaler",
]
