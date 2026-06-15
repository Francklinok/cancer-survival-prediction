"""
dataset_export.py — Saving and exporting pipeline datasets and results.

Functions:
  - save_dataset(df, path, formats)                           → list[str]
  - save_model(model, path, model_name)                       → str
  - save_pipeline_results(state, save_path, formats)         → list[str]
  - save_reconstructed_dataset(results, data, target, path)  → (DataFrame, list[str])
  - load_dataset(path)                                        → pd.DataFrame
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger("ml_framework.dataset_export")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET SAVING
# ──────────────────────────────────────────────────────────────────────────────


def save_dataset(
    df: pd.DataFrame,
    path: Union[str, Path],
    formats: Tuple[str, ...] = ("csv",),
    filename: str = "dataset",
    index: bool = False,
) -> List[str]:
    """
    Save a DataFrame in one or more formats.

    Parameters
    ----------
    df       : DataFrame to save
    path     : destination directory
    formats  : tuple of formats — 'csv', 'parquet', 'excel', 'pickle', 'json'
    filename : base file name (without extension)
    index    : include the index in the output file

    Returns
    -------
    list of created file paths
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for fmt in formats:
        fmt = fmt.lower().strip()
        try:
            if fmt == "csv":
                fp = save_dir / f"{filename}.csv"
                df.to_csv(fp, index=index)
                saved.append(str(fp))
            elif fmt in ("parquet", "feather"):
                fp = save_dir / f"{filename}.parquet"
                df.to_parquet(fp, index=index)
                saved.append(str(fp))
            elif fmt in ("excel", "xlsx"):
                fp = save_dir / f"{filename}.xlsx"
                df.to_excel(fp, index=index)
                saved.append(str(fp))
            elif fmt in ("pickle", "pkl"):
                fp = save_dir / f"{filename}.pkl"
                df.to_pickle(fp)
                saved.append(str(fp))
            elif fmt == "json":
                fp = save_dir / f"{filename}.json"
                df.to_json(fp, orient="records", indent=2, force_ascii=False)
                saved.append(str(fp))
            else:
                logger.warning("Unknown save format: '%s'", fmt)
        except Exception as exc:
            logger.error("Error saving as '%s': %s", fmt, exc)

    logger.info("Dataset saved: %d file(s) → %s", len(saved), path)
    for fp in saved:
        print(f"{fp}")

    return saved


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SAVING
# ──────────────────────────────────────────────────────────────────────────────


def save_model(
    model: Any,
    path: Union[str, Path],
    model_name: str = "model",
) -> str:
    """
    Save a sklearn model (via joblib if available, else pickle).

    Parameters
    ----------
    model      : trained sklearn estimator
    path       : destination directory
    model_name : file name (without extension)

    Returns
    -------
    path of the created file
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    fp = save_dir / f"{model_name}.pkl"

    try:
        import joblib
        joblib.dump(model, fp)
    except ImportError:
        with open(fp, "wb") as f:
            pickle.dump(model, f)

    logger.info("Model '%s' saved → %s", model_name, fp)
    print(f" Model saved: {fp}")
    return str(fp)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE RESULTS SAVING
# ──────────────────────────────────────────────────────────────────────────────


def save_pipeline_results(
    state: Dict[str, Any],
    save_path: Union[str, Path] = "./results/",
    formats: Tuple[str, ...] = ("csv", "model", "pickle"),
    target_column: str = "target",
) -> List[str]:
    """
    Save the complete results of a MedicalMLPipeline run.

    Saves:
      - final_dataset.csv    (if 'csv' in formats)
      - <model_name>.pkl     (if 'model' in formats, one per model)
      - complete_state.pkl   (if 'pickle' in formats)

    Parameters
    ----------
    state        : pipeline state dict (self.state)
    save_path    : destination directory
    formats      : formats to save
    target_column: name of the target column (excluded from X)

    Returns
    -------
    list of created file paths
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    print(f"\n Saving results → {save_dir}")

    # Final dataset
    if "csv" in formats and state.get("final_dataset") is not None:
        fps = save_dataset(
            state["final_dataset"], save_dir,
            formats=("csv",), filename="final_dataset",
        )
        saved.extend(fps)

    # Models
    if "model" in formats and state.get("models"):
        models_dir = save_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for model_name, model_obj in state["models"].items():
            fp = save_model(model_obj, models_dir, model_name)
            saved.append(fp)

    # Full state dump (serializable)
    if "pickle" in formats:
        try:
            import joblib
            fp = save_dir / "complete_state.pkl"
            serializable_state = {
                k: v for k, v in state.items()
                if isinstance(v, (str, int, float, bool, dict, list, pd.DataFrame, pd.Series))
                   or hasattr(v, "__sklearn_tags__")  # sklearn estimators
            }
            joblib.dump(serializable_state, fp)
            saved.append(str(fp))
            print(f" Complete state: {fp}")
        except Exception as exc:
            logger.error("Error dumping pipeline state: %s", exc)

    print(f" {len(saved)} file(s) saved.")
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTED DATASET EXPORT
# ──────────────────────────────────────────────────────────────────────────────


def save_reconstructed_dataset(
    results: Dict[str, Any],
    data: pd.DataFrame,
    target_column: str = "target",
    save_path: Union[str, Path] = "./reconstructed_dataset/",
    save_formats: Tuple[str, ...] = ("csv", "pickle"),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reconstruct a dataset from selected features and save it.

    Parameters
    ----------
    results       : selection results dict (key 'top_features_combines' or 'features')
    data          : full source DataFrame
    target_column : target column to include
    save_path     : destination directory
    save_formats  : save formats ('csv', 'pickle')

    Returns
    -------
    (df_full, saved_files)
    """
    # Retrieve selected features
    selected_features = (
        results.get("top_features_combines")
        or results.get("features")
        or results.get("significant_features")
        or []
    )

    if not selected_features:
        logger.warning("No selected features found in results.")
        selected_features = [c for c in data.columns if c != target_column]

    # Remove target if present in features list
    selected_features = [f for f in selected_features if f != target_column]

    # Filter to available columns
    available = [f for f in selected_features if f in data.columns]
    missing   = set(selected_features) - set(available)
    if missing:
        logger.warning("Features missing from data: %s", missing)

    X_final = data[available].copy()
    y_final = data[target_column].copy() if target_column in data.columns else pd.Series(dtype=float)

    df_full = X_final.copy()
    if not y_final.empty:
        df_full[target_column] = y_final

    saved_files: List[str] = []
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in save_formats:
        fps = save_dataset(df_full, save_dir, formats=("csv",), filename="dataset_selected")
        saved_files.extend(fps)

    if "pickle" in save_formats:
        fp = save_dir / "dataset_selected.pkl"
        payload = {
            "X_final": X_final,
            "y_final": y_final,
            "selected_features": available,
            "results": results,
        }
        with open(fp, "wb") as f:
            pickle.dump(payload, f)
        saved_files.append(str(fp))
        print(f"Pickle: {fp}")

    print(f" Reconstructed dataset ({df_full.shape}) saved → {save_path}")
    return df_full, saved_files


# ──────────────────────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a dataset from a file (CSV, Parquet, Excel, Pickle, JSON).

    Parameters
    ----------
    path : path to the file

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(p)
    elif ext in (".parquet", ".feather"):
        return pd.read_parquet(p)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(p)
    elif ext == ".json":
        return pd.read_json(p)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'")
