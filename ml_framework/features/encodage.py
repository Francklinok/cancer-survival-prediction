"""
encodage.py — Encoding facade for the features module.

Re-exports the main encoding functions from ml_framework.preprocessing.encoding.

Recommended import:
    from ml_framework.features.encodage import encode_dataframe, target_encoding
"""

from ml_framework.preprocessing.encoding import (
    encode_dataframe,
    encodage_processing,  # backward-compat alias
    frequency_encoding,
    one_hot_encode,
    target_encoding,
)

__all__ = [
    "encode_dataframe",
    "encodage_processing",
    "target_encoding",
    "frequency_encoding",
    "one_hot_encode",
]
