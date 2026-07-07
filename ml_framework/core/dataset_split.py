"""
dataset_split.py — Dataset splitting utilities.

Provides robust functions to split data into train / validation / test sets with:
  - Support for tabular data (DataFrame or array)
  - Automatic stratification for classification tasks
  - Temporal split for sequential data
  - K-Fold, Stratified K-Fold, Group K-Fold

Public functions:
  - split_train_test(df, target, test_size, val_size, stratify, random_state)
  - split_temporal(df, date_column, test_ratio, val_ratio)
  - make_cv_splitter(cv, stratified, groups, random_state)
  - validate_split_sizes(n_samples, test_size, val_size, min_samples)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    train_test_split,
)

logger = logging.getLogger("ml_framework.dataset_split")


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────


def validate_split_sizes(
    n_samples: int,
    test_size: float = 0.2,
    val_size: float = 0.0,
    min_samples: int = 10,
) -> None:
    """
    Verify that split proportions are valid.

    Parameters
    ----------
    n_samples   : total number of observations
    test_size   : test set proportion (0 < test_size < 1)
    val_size    : validation set proportion (0 ≤ val_size < 1)
    min_samples : minimum number of observations per split

    Raises
    ------
    ValueError if proportions are invalid or the dataset is too small.
    """
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, received: {test_size}")
    if not (0 <= val_size < 1):
        raise ValueError(f"val_size must be between 0 and 1, received: {val_size}")
    if test_size + val_size >= 1:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) must be < 1."
        )

    n_test = int(n_samples * test_size)
    n_val  = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val

    if n_train < min_samples:
        raise ValueError(
            f"Training set too small ({n_train} < {min_samples}). "
            f"Reduce test_size or val_size."
        )
    if n_test < min_samples:
        raise ValueError(
            f"Test set too small ({n_test} < {min_samples}). "
            f"Increase test_size or dataset size."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT (+ OPTIONAL VALIDATION)
# ──────────────────────────────────────────────────────────────────────────────

def split_train_test(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify: bool = True,
    random_state: int = 42,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series],
]:
    """
    Split a DataFrame into train, test (and optional validation) sets.

    Parameters
    ----------
    df           : full DataFrame (features + target)
    target       : name of the target column
    test_size    : test set proportion
    val_size     : validation set proportion (0 = no validation set)
    stratify     : stratify by target (for classification)
    random_state : random seed for reproducibility

    Returns
    -------
    Without validation : (X_train, X_test, y_train, y_test)
    With validation    : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    validate_split_sizes(len(df), test_size, val_size)

    X = df.drop(columns=[target])
    y = df[target]

    strat_arr = y if stratify and y.nunique() <= 20 else None

    # First split: train+val vs test
    try:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=strat_arr,
            random_state=random_state,
        )
    except ValueError:
        # Stratification can fail (e.g. a class with too few members) —
        # fall back to an unstratified split rather than raising.
        logger.warning("Stratified split failed — falling back to unstratified split.")
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    if val_size == 0.0:
        logger.info(
            "Train/test split: train=%d | test=%d | stratified=%s",
            len(X_trainval), len(X_test), stratify,
        )
        return X_trainval, X_test, y_trainval, y_test

    # Second split: train vs val
    val_ratio_adjusted = val_size / (1 - test_size)
    strat_val = y_trainval if stratify and y_trainval.nunique() <= 20 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        stratify=strat_val,
        random_state=random_state,
    )

    logger.info(
        "Train/val/test split: train=%d | val=%d | test=%d | stratified=%s",
        len(X_train), len(X_val), len(X_test), stratify,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL SPLIT
# ──────────────────────────────────────────────────────────────────────────────


def split_temporal(
    df: pd.DataFrame,
    date_column: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0,
    target: Optional[str] = None,
) -> dict:
    """
    Split a time-series DataFrame into train / val / test preserving chronological order.

    Parameters
    ----------
    df          : DataFrame with a date column
    date_column : name of the date/datetime column
    test_ratio  : final proportion reserved for the test set
    val_ratio   : proportion reserved for validation (between train and test)
    target      : target column name (if provided, also returns y_train, y_val, y_test)

    Returns
    -------
    dict with keys: df_train, df_val (optional), df_test,
                    X_train, X_test, y_train, y_test (if target provided)
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame.")

    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)

    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_test - n_val

    df_train = df_sorted.iloc[:n_train]
    df_val   = df_sorted.iloc[n_train:n_train + n_val] if n_val > 0 else pd.DataFrame()
    df_test  = df_sorted.iloc[n_train + n_val:]

    logger.info(
        "Temporal split: train=%d | val=%d | test=%d",
        len(df_train), len(df_val), len(df_test),
    )

    result = {"df_train": df_train, "df_val": df_val, "df_test": df_test}

    if target and target in df.columns:
        result["X_train"] = df_train.drop(columns=[target])
        result["y_train"] = df_train[target]
        if not df_val.empty:
            result["X_val"] = df_val.drop(columns=[target])
            result["y_val"] = df_val[target]
        result["X_test"] = df_test.drop(columns=[target])
        result["y_test"] = df_test[target]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CV SPLITTER FACTORY
# ──────────────────────────────────────────────────────────────────────────────


def make_cv_splitter(
    cv: int = 5,
    stratified: bool = True,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
):
    """
    Return an appropriate scikit-learn cross-validation splitter.

    Parameters
    ----------
    cv           : number of folds
    stratified   : use StratifiedKFold (recommended for classification)
    groups       : group array (for GroupKFold)
    random_state : random seed

    Returns
    -------
    sklearn splitter (StratifiedKFold | GroupKFold | KFold)
    """
    if groups is not None:
        logger.info("CV: GroupKFold (k=%d)", cv)
        return GroupKFold(n_splits=cv)
    elif stratified:
        logger.info("CV: StratifiedKFold (k=%d)", cv)
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        logger.info("CV: KFold (k=%d)", cv)
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
