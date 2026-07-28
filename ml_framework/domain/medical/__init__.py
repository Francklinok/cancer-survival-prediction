"""
ml_framework.domain.medical — Optional domain pack for the oncology /
cancer-recurrence reference dataset this framework was originally built
against (see notbook/test_recur.ipynb).

The core framework (ml_framework.*) is domain-neutral: it works on any
tabular dataset out of the box, with no medical vocabulary or column names
baked into its defaults. 

This package is where that original domain knowledge lives instead: column 
mappings, engineered clinical features, and risk-reporting thresholds — keeping 
them available in a single import without constraining the generic API.

Usage
-----
    from ml_framework.config.config import FrameworkConfig
    config = FrameworkConfig.from_domain("medical")   # column-name defaults

    from ml_framework.domain.medical import get_encoding_mappings, add_oncology_features
    binary_map, ordinal_map, nominal_cols = get_encoding_mappings()
    df_encoded, report = encode_dataframe(df, binary_mappings=binary_map,
                                           ordinal_mappings=ordinal_map,
                                           nominal_columns=nominal_cols)
"""

from ml_framework.domain.medical.encoding_mappings import get_encoding_mappings
from ml_framework.domain.medical.oncology_features import add_oncology_features

__all__ = ["get_encoding_mappings", "add_oncology_features"]
