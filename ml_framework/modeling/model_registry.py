"""
model_registry.py — Classification / regression model registry.

Provides:
  - get_models()   : dict of pre-configured base models
  - get_model()    : single-model access by name prefix
  - list_models()  : list of available model names
  - MODEL_REGISTRY : centralized configuration reference

Models included:
  RandomForest, GradientBoosting, XGBoost, LightGBM,
  LogisticRegression, SVM, KNN, DecisionTree, ExtraTree,
  AdaBoost, BaggingClassifier
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger("ml_framework.model_registry")


# =============================================================================
# MODEL REGISTRY
# =============================================================================


def get_models(
    random_state: int = 42,
    n_jobs: int = -1,
    subset: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Return a dictionary of ready-to-use sklearn models.

    Parameters
    ----------
    random_state : reproducibility seed
    n_jobs       : parallel workers (-1 = all CPUs)
    subset       : list of model name prefixes to include
                   ('rf', 'gb', 'lr', 'svm', 'knn', 'dt', 'et', 'ada', 'bag',
                    'xgb', 'lgb').
                   If None → all models are returned.

    Returns
    -------
    dict {model_name: estimator}
    """
    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        _xgb = XGBClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        has_xgb = True
    except ImportError:
        has_xgb = False

    # Optional LightGBM
    try:
        import lightgbm as lgb
        _lgb = lgb.LGBMClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
        has_lgb = True
    except ImportError:
        has_lgb = False

    registry: Dict[str, object] = {
        "rf_default": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=n_jobs
        ),
        "gb_default": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
        "lr_default": LogisticRegression(
            random_state=random_state, max_iter=1_000, solver="lbfgs",
            multi_class="auto", n_jobs=n_jobs,
        ),
        "svm_default": SVC(
            random_state=random_state, probability=True, kernel="rbf"
        ),
        "knn_default": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        "dt_default":  DecisionTreeClassifier(random_state=random_state, max_depth=8),
        "et_default":  ExtraTreesClassifier(
            n_estimators=100, random_state=random_state, n_jobs=n_jobs
        ),
        "ada_default": AdaBoostClassifier(
            n_estimators=100, random_state=random_state
        ),
        "bag_default": BaggingClassifier(
            n_estimators=50, random_state=random_state, n_jobs=n_jobs
        ),
    }

    if has_xgb:
        registry["xgb_default"] = _xgb
    if has_lgb:
        registry["lgb_default"] = _lgb

    if subset:
        # Accept both full names and short prefixes
        _name_map = {
            "randomforest":       "rf",
            "gradientboosting":   "gb",
            "logisticregression": "lr",
            "svm":                "svm",
            "supportvector":      "svm",
            "knn":                "knn",
            "knearestneighbors":  "knn",
            "decisiontree":       "dt",
            "extratrees":         "et",
            "adaboost":           "ada",
            "bagging":            "bag",
            "xgboost":            "xgb",
            "lightgbm":           "lgb",
        }
        normalized_subset: set = set()
        for name in subset:
            n_low  = name.lower().replace(" ", "").replace("_", "").replace("classifier", "")
            prefix = _name_map.get(n_low, n_low)
            normalized_subset.add(prefix)

        registry = {
            k: v for k, v in registry.items()
            if k.split("_")[0] in normalized_subset
        }

    logger.info("Registry loaded: %d model(s) available.", len(registry))
    return registry


def get_model(name: str, random_state: int = 42) -> object:
    """Return a single model by its prefix (e.g. 'rf', 'gb', 'lr')."""
    models = get_models(random_state=random_state, subset=[name])
    key    = f"{name}_default"
    if key not in models:
        raise ValueError(
            f"Model '{name}' not found. Available: {list_models()}"
        )
    return models[key]


def list_models() -> List[str]:
    """Return the list of available model name prefixes."""
    return ["rf", "gb", "lr", "svm", "knn", "dt", "et", "ada", "bag", "xgb", "lgb"]
