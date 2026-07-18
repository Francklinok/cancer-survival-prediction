"""
profiling.py — Builds the DatasetProfile that routes Decision Engine rules.

"""

from __future__ import annotations

import pandas as pd

from ml_framework.orchestration.v2.contracts import DatasetProfile


def build_dataset_profile(df: pd.DataFrame) -> DatasetProfile:
    """
    df is expected to be ctx.df_work at the point profiling/EDA has run —
    i.e. after ingest, since problem_type/dimensions describe the working
    dataset, not the raw file.
    """
    n_rows, n_columns = df.shape
    dtype_counts = df.dtypes.astype(str).value_counts()
    dominant_dtype = dtype_counts.index[0] if not dtype_counts.empty else ""

    return DatasetProfile(
        problem_type="tabular",
        n_rows=n_rows,
        n_columns=n_columns,
        dominant_dtype=dominant_dtype,
        metadata={"dtype_counts": dtype_counts.to_dict()},
    )
