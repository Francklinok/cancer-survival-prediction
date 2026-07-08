"""
model_explainability.py — Advanced ML model interpretability.

Methods implemented:
  - SHAP (SHapley Additive exPlanations): global + local
      • summary plot, bar plot, beeswarm, dependence plots
      • interaction values (TreeSHAP)
      • force plot for individual cases
  - LIME (Local Interpretable Model-agnostic Explanations)
      • local explanations for specific instances
  - Partial Dependence Plots (PDP + ICE)

Automatic scientific interpretations are generated.
SHAP visualizations delegated to visualization.interpretability.shap_plots.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_framework.visualization.interpretability.shap_plots import (
    plot_shap_summary,
    plot_shap_dependence,
)
from ml_framework.visualization.base import section_header

logger = logging.getLogger("ml_framework.explainability")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with only float-convertible columns.

    Drops columns whose values cannot be cast to float — typically columns
    that contain string-serialised vectors such as '[7.89E-3,9.57E-3,...]'
    produced by embedding steps that forgot to expand list outputs into
    individual numeric columns.
    """
    bad: list = []
    for col in df.columns:
        try:
            df[col].astype(float)
        except (ValueError, TypeError):
            bad.append(col)
    if bad:
        logger.warning(
            "_to_numeric: dropping %d non-numeric column(s): %s", len(bad), bad
        )
    return df.drop(columns=bad).astype(float).fillna(0)


def _patch_xgb_feature_names(model, feature_names: list) -> object:
    """
    XGBoost stores feature names from training inside the Booster object.
    If those names contain string-encoded vectors, SHAP's TreeExplainer
    fails when it reads the Booster metadata.

    This function returns a patched copy of the model whose internal Booster
    has clean integer-indexed feature names that match the sanitised X_test.
    The original model object is NOT modified.
    """
    try:
        import copy
        import json as _json

        import xgboost as xgb

        if not isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            return model

        # Serialise → patch → deserialise via booster JSON round-trip
        booster = model.get_booster()
        model_json = booster.save_raw(raw_format="json").decode("utf-8")
        cfg = _json.loads(model_json)

        # Replace feature names with clean sequential strings f0, f1, ...
        clean_names = [f"f{i}" for i in range(len(feature_names))]

        def _replace_names(obj):
            if isinstance(obj, dict):
                if "feature_names" in obj:
                    obj["feature_names"] = clean_names
                for v in obj.values():
                    _replace_names(v)
            elif isinstance(obj, list):
                for item in obj:
                    _replace_names(item)

        _replace_names(cfg)

        patched_booster = xgb.Booster()
        patched_booster.load_model(bytearray(_json.dumps(cfg).encode("utf-8")))

        patched_model = copy.copy(model)
        patched_model._Booster = patched_booster

        # Return both the patched model and the clean feature names
        return patched_model, clean_names

    except Exception as exc:
        logger.warning("_patch_xgb_feature_names failed (%s) — using numpy fallback.", exc)
        return model, feature_names


# =============================================================================
# SHAP INTERPRETATION
# =============================================================================


def interpret_model_with_shap(
    model,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    plot_type: str = "summary",
    n_dependence_plots: int = 3,
    verbose: bool = True,
) -> Optional[object]:
    """
    Global and local model interpretation using SHAP.

    Parameters
    ----------
    model              : trained model (sklearn / xgboost / lightgbm compatible)
    X_test             : test data
    feature_names      : feature names
    max_display        : max features shown in plots
    plot_type          : 'summary' | 'bar' | 'beeswarm' | 'all'
    n_dependence_plots : number of dependence plots for top-N features
    verbose            : print textual interpretation

    Returns
    -------
    shap.Explainer instance, or None on failure
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed: pip install shap")
        return None

    section_header("SHAP INTERPRETATION")

    if feature_names is None and hasattr(X_test, "columns"):
        feature_names = X_test.columns.tolist()

    X_test = _to_numeric(X_test)
    if feature_names is not None:
        # Re-align feature_names after potential column drops
        feature_names = [f for f in feature_names if f in X_test.columns]
    else:
        feature_names = X_test.columns.tolist()

    model_name = model.__class__.__name__.lower()
    is_xgb     = "xgb" in model_name
    is_tree    = any(k in model_name for k in
                     ("forest", "gradient", "tree", "xgb", "lgb", "extra", "catboost"))
    is_linear  = "linear" in model_name or "logistic" in model_name

    explainer   = None
    shap_values = None
    shap_feat_names = list(feature_names)
    X_shap = X_test.copy()

    # XGBoost: patch internal Booster feature names before handing off to TreeExplainer
    if is_xgb:
        try:
            patched_model, clean_names = _patch_xgb_feature_names(model, feature_names)
            # Rename X columns to match patched booster (f0, f1, ...)
            X_shap_clean = X_shap.copy()
            X_shap_clean.columns = clean_names
            explainer   = shap.TreeExplainer(patched_model)
            shap_values = explainer.shap_values(X_shap_clean)
            shap_feat_names = feature_names   # restore original names for display
            logger.info("SHAP: XGBoost patched TreeExplainer succeeded.")
        except Exception as exc:
            logger.warning("XGB patched TreeExplainer failed (%s) — trying numpy array.", exc)
            explainer   = None
            shap_values = None

    # XGBoost fallback: raw numpy array bypasses the internal feature-name check
    if is_xgb and shap_values is None:
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap.values)   # numpy, no column names
            logger.info("SHAP: XGBoost numpy-array TreeExplainer succeeded.")
        except Exception as exc:
            logger.warning("XGB numpy TreeExplainer failed (%s).", exc)
            explainer   = None
            shap_values = None

    if not is_xgb and is_tree and shap_values is None:
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
        except Exception as exc:
            logger.warning("TreeExplainer.shap_values() failed: %s", exc)
            explainer   = None
            shap_values = None

    if shap_values is None and is_linear:
        try:
            explainer   = shap.LinearExplainer(model, X_shap)
            shap_values = explainer.shap_values(X_shap)
        except Exception as exc:
            logger.warning("LinearExplainer failed: %s", exc)
            explainer   = None
            shap_values = None

    # All SHAP strategies failed — fall back to permutation importance (model-agnostic)
    if shap_values is None:
        try:
            logger.info("SHAP: all explainers failed — using permutation importance fallback.")
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import f1_score

            perm = permutation_importance(
                model, X_shap, model.predict(X_shap),
                scoring="f1_macro", n_repeats=10,
                random_state=42, n_jobs=-1,
            )
            imp_mean = perm.importances_mean
            imp_std  = perm.importances_std

            sorted_idx = np.argsort(imp_mean)[::-1][:max_display]
            top_names  = [shap_feat_names[i] for i in sorted_idx if i < len(shap_feat_names)]
            top_imp    = imp_mean[sorted_idx]
            top_std    = imp_std[sorted_idx]

            plt.figure(figsize=(10, max(4, len(top_names) * 0.4)))
            plt.barh(top_names[::-1], top_imp[::-1],
                     xerr=top_std[::-1], color="steelblue", alpha=0.8)
            plt.xlabel("Permutation importance (F1-macro drop)")
            plt.title(f"Feature importance (permutation) — top {max_display} features")
            plt.tight_layout()
            plt.show()

            # CRITICAL MISMATCH: Permutation Importance != SHAP Values
            # We cannot use permutation importance to fake a SHAP matrix. 
            # 
            # Why? Permutation importance only tells us *how much* a feature matters 
            # (e.g., "Age is important"), but it has no direction. It cannot tell us 
            # *how* it matters (e.g., "Higher age increases risk").
            # 
            # If we force this into the SHAP interpreter, the code will literally 
            # invent directional claims ("↑ increases risk") out of thin air.
            # 
            # Safe fallback: Return None. Functions calling this must handle the 
            # absence of SHAP values explicitly.

            logger.warning(
                "SHAP: falling back to permutation importance (no SHAP values available). "
                "The permutation importance plot is shown above. "
                "Direction of effect per feature is NOT available — run SHAP for that."
            )
            print(
                "\n  All SHAP explainers failed. Showing permutation importance instead.\n"
                "     Permutation importance ≠ Shapley values: no direction, no interaction.\n"
                "     For SHAP-based interpretation, ensure the model is tree- or linear-based\n"
                "     and that the 'shap' package is installed.\n"
            )
            return None   # no SHAP explainer object to return

        except Exception as exc:
            logger.error("Permutation importance fallback also failed: %s", exc)
            return None

    # TreeExplainer multiclass returns list[n_classes] of (n_samples, n_features);
    if isinstance(shap_values, list):
        shap_arr    = np.array(shap_values)          # (n_classes, n_samples, n_features)
        shap_matrix = np.mean(np.abs(shap_arr), axis=0)   # (n_samples, n_features)
        shap_list   = shap_values
    elif hasattr(shap_values, "values"):
        raw = shap_values.values
        if raw.ndim == 3:
            shap_matrix = np.mean(np.abs(raw), axis=2)
            shap_list   = [raw[:, :, i] for i in range(raw.shape[2])]
        else:
            shap_matrix = raw
            shap_list   = None
    else:
        # Newer shap versions have TreeExplainer.shap_values() return a bare
        # ndarray shaped (n_samples, n_features, n_classes) for multiclass/
        # binary classifiers, instead of the old list[n_classes] or an
        # Explanation object — treating it as already-2D silently corrupted
        # the feature axis (mean over samples instead of over classes).
        raw = np.asarray(shap_values, dtype=float)
        if raw.ndim == 3:
            shap_matrix = np.mean(np.abs(raw), axis=2)
            shap_list   = [raw[:, :, i] for i in range(raw.shape[2])]
        else:
            shap_matrix = raw
            shap_list   = None

    # Align column count in case _to_numeric dropped some features
    n_feat = len(shap_feat_names)
    if shap_matrix.shape[1] != n_feat:
        shap_matrix = shap_matrix[:, :n_feat]
        if shap_list:
            shap_list = [sv[:, :n_feat] for sv in shap_list]

    class_labels = getattr(model, "classes_", None)
    X_plot = pd.DataFrame(X_shap.values if hasattr(X_shap, "values") else X_shap,
                          columns=shap_feat_names)

    # Delegate to plot_shap_summary rather than hand-rolling a duplicate
    # bar/beeswarm implementation here — keeps figsize/plot_size and any
    # future styling improvements in one place instead of two divergent copies.
    try:
        plot_shap_summary(
            shap_matrix, X_plot, shap_feat_names,
            max_display=max_display, plot_type="bar",
        )
    except Exception as exc:
        logger.warning("Global SHAP bar plot failed: %s", exc)

    if shap_list is not None:
        for cls_idx, sv_cls in enumerate(shap_list):
            cls_name = (str(class_labels[cls_idx])
                        if class_labels is not None and cls_idx < len(class_labels)
                        else f"Class {cls_idx}")
            try:
                sv_plot = sv_cls[:, :n_feat]
                plot_shap_summary(
                    sv_plot, X_plot, shap_feat_names,
                    max_display=max_display, plot_type="summary",
                    title_suffix=f" — {cls_name}",
                )
            except Exception as exc:
                logger.warning("Beeswarm class %d failed: %s", cls_idx, exc)
    else:
        try:
            plot_shap_summary(
                shap_matrix, X_plot, shap_feat_names,
                max_display=max_display, plot_type="summary",
            )
        except Exception as exc:
            logger.warning("Beeswarm failed: %s", exc)

    if n_dependence_plots > 0:
        dep_matrix = shap_list[0][:, :n_feat] if shap_list else shap_matrix
        try:
            plot_shap_dependence(dep_matrix, X_plot, shap_feat_names, n_dependence_plots)
        except Exception as exc:
            logger.warning("Dependence plots failed: %s", exc)

    if verbose:
        _interpret_shap_results(shap_matrix, shap_feat_names, max_display)

    return explainer


def _interpret_shap_results(
    shap_matrix: np.ndarray,
    feature_names: List[str],
    top_n: int,
) -> None:
    """Generate a textual interpretation of SHAP results."""
    section_header("SHAP VALUES INTERPRETATION")

    mean_abs   = np.abs(shap_matrix).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    print("  Most influential variables on the prediction:")
    for rank, idx in enumerate(sorted_idx[:min(top_n, len(feature_names))], 1):
        if idx >= len(feature_names):
            continue
        fname      = feature_names[idx]
        importance = mean_abs[idx]
        mean_signed = shap_matrix[:, idx].mean()
        direction   = "↑ increases" if mean_signed > 0 else "↓ decreases"
        print(f"  {rank:>3}. {fname:<35} |shap|={importance:.4f}  ({direction} the risk on average)")

    print("\n  Scientific note:")
    print("    • Positive SHAP values increase the probability of the positive class.")
    print("    • Magnitude indicates the strength of the effect.")
    print("    • Color (red=high feature value, blue=low) shows the direction.")


# =============================================================================
# LIME
# =============================================================================


def interpret_model_with_lime(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 5,
    num_features: int = 10,
) -> Optional[object]:
    """
    Local model interpretation using LIME.

    Explains individual predictions via local linear approximation.

    Parameters
    ----------
    model         : trained model
    X_train       : training data (used to fit the LIME explainer)
    X_test        : test data (instances to explain)
    feature_names : feature names
    class_names   : class label names
    num_samples   : number of instances to explain
    num_features  : number of features in each LIME explanation

    Returns
    -------
    lime.lime_tabular.LimeTabularExplainer, or None on failure
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        logger.error("LIME not installed: pip install lime")
        return None

    section_header("LIME INTERPRETATION")

    if feature_names is None and hasattr(X_train, "columns"):
        feature_names = X_train.columns.tolist()

    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    categorical_features = (
        [i for i, dtype in enumerate(X_train.dtypes)
         if dtype == object or str(dtype) == "category"]
        if hasattr(X_train, "dtypes") else []
    )

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values if hasattr(X_train, "values") else X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        categorical_features=categorical_features,
    )

    predict_fn    = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    sample_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices, 1):
        instance = X_test.iloc[idx].values if hasattr(X_test, "iloc") else X_test[idx]

        try:
            # Explain the class the model actually predicted for this instance
            # — hardcoding label=1 would silently show a different class's
            # explanation than the one printed as "Prediction" below whenever
            # the model predicts anything other than class 1 (always true in
            # genuinely multiclass settings, and half the time even in binary).
            predicted_label = int(model.predict([instance])[0])

            exp = explainer.explain_instance(
                instance, predict_fn, num_features=num_features, labels=(predicted_label,)
            )

            print(f"\n  Instance {i} (index={idx}):")
            print(f"  Prediction: class {predicted_label}")
            print("  Top local features:")
            for feat, weight in exp.as_list(label=predicted_label)[:5]:
                direction = "+" if weight > 0 else "-"
                print(f"    {direction} {feat:<40} {weight:+.4f}")

            fig = exp.as_pyplot_figure(label=predicted_label)
            plt.title(f"LIME — Instance {i}", fontweight="bold")
            plt.tight_layout()
            plt.show()

        except Exception as exc:
            logger.warning("LIME instance %d failed: %s", i, exc)

    return explainer


# =============================================================================
# PARTIAL DEPENDENCE PLOTS
# =============================================================================


def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    features: Optional[List[Union[str, int]]] = None,
    top_n: int = 6,
    kind: str = "average",
) -> None:
    """
    Plot Partial Dependence Plots (PDP) or Individual Conditional Expectation (ICE).

    Parameters
    ----------
    model    : trained sklearn-compatible model
    X        : feature data
    features : feature names or indices to plot (auto-selects top_n by importance if None)
    top_n    : number of features when auto-selecting
    kind     : 'average' (PDP) | 'individual' (ICE) | 'both'
    """
    try:
        from sklearn.inspection import PartialDependenceDisplay
    except ImportError:
        logger.error("PartialDependenceDisplay not available — sklearn >= 1.0 required.")
        return

    if features is None:
        if hasattr(model, "feature_importances_"):
            top_idx  = np.argsort(model.feature_importances_)[-top_n:][::-1]
            features = [int(i) for i in top_idx]
        else:
            features = list(range(min(top_n, X.shape[1])))

    try:
        fig, ax = plt.subplots(figsize=(16, 4 * ((len(features) + 2) // 3)))
        PartialDependenceDisplay.from_estimator(
            model, X, features,
            kind=kind,
            n_cols=3,
            ax=ax,
            random_state=42,
        )
        fig.suptitle(
            f"Partial Dependence Plots ({kind}) — Top {top_n} Features",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        logger.error("PDP failed: %s", exc)
