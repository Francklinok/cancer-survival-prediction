# ml_framework

A general-purpose, production-minded ML framework for tabular data — data profiling, cleaning, feature engineering, model training with nested cross-validation, explainability (SHAP/LIME), fairness auditing, and drift monitoring, all wired together by an adaptive pipeline orchestrator.

It's domain-neutral by default: point it at any CSV with a target column and it works. It was originally built and validated against an oncology reference dataset (cancer recurrence prediction), and that domain knowledge is still available as an optional pack — see [Domain packs](#domain-packs) below — but nothing in the core framework assumes medical data anymore.

---

## ⚠️ About the reference dataset

The example dataset (`ml_framework/data/cancer issue.csv`) and the reference notebook (`notebook/test_recur.ipynb`) are **synthetic** — they don't represent real patients or real clinical outcomes. They exist to give the framework a full end-to-end workout (profiling through explainability) on realistic-looking tabular data, and to document, honestly, what a rigorous analysis of this kind of dataset looks like — including the finding that no statistically significant signal was detected between any feature and the target on this particular synthetic data. Nothing here should be read as medical guidance or used for real clinical decisions.

---

## What's actually in here

- Data profiling, cleaning, and quality scoring
- Missing-value handling (MICE / KNN / MissForest / simple), outlier detection and treatment
- Categorical encoding with a domain-neutral fallback (auto one-hot for anything not explicitly mapped)
- Adaptive feature engineering — candidate features are only kept if they measurably improve cross-validated performance
- Zero-leakage normalization (scalers fit on train only, applied to test after)
- Multi-model training, hyperparameter tuning (Randomized/Grid/Bayesian), nested cross-validation, stacking ensembles
- SHAP + LIME explainability, permutation importance, partial dependence plots
- Fairness auditing (statistical parity, equal opportunity, 80% rule) across any sensitive attribute you specify
- Automatic Model Card generation (Google 2019 standard)
- Data drift monitoring (PSI, KS test, Wasserstein distance) for production use
- Two pipeline orchestrators — see [Orchestration](#orchestration) — that wire all of the above into a single adaptive pipeline instead of requiring you to call each piece by hand

---

## Quick start

```python
from ml_framework.orchestration.v2.facade import MLPipelineV2
from ml_framework.config.config import FrameworkConfig

pipeline = MLPipelineV2()
pipeline.run("your_data.csv", target_column="target")

# Before trusting anything else, check whether the run actually went well:
summary = pipeline.get_summary()
print(summary["success"])              # False if any module failed
print(summary["validation_problems"])  # sanity checks that don't just trust each module's own bookkeeping
print(summary["data_quality_warnings"])# e.g. severe class imbalance, leakage-flagged features

model   = pipeline.get_best_model()
card    = pipeline.context.model_card
metrics = summary["final_metrics"]
```

Run only part of the pipeline:

```python
pipeline.run("your_data.csv", target_column="target",
             steps=["ingest", "clean", "eda", "missing"])
```

Attach monitoring without touching the pipeline itself:

```python
def log_shape(ctx):
    print(f"  Shape after step: {ctx.df_work.shape}")

pipeline.add_hook("clean", "after", log_shape)
pipeline.run("your_data.csv", target_column="target")
```

---

## Orchestration

There are two orchestrators in this repo, and that's intentional, not leftover clutter:

- **`MLPipelineV2`** (`ml_framework/orchestration/v2/`) — the one to use. It plans which steps a given dataset actually needs (skips a missing-value step on data with no missing values, for instance), validates its own dependency graph before running anything, and — this is the part that matters most if you're pointing it at a dataset the framework hasn't seen before — checks its own results after the fact instead of just trusting each module's bookkeeping. `get_summary()["validation_problems"]` catches things like train/test row counts not adding up, or train/test index overlap, before you find out the hard way that a metric was computed on the wrong data.
- **`MLPipeline`** (`ml_framework/orchestration/pipeline.py`, the original v1) — kept as the reference implementation v2 is tested against. A non-regression test runs both engines on the real reference dataset and asserts their metrics match exactly. It's the actual proof that v2 is correct, not just that it runs.

Read [`docs/orchestration_v2_architecture.md`](docs/orchestration_v2_architecture.md) for how the planning/execution split actually works, and why a couple of things (a distributed event bus, plugin auto-discovery) were deliberately left out.

### How do I know the results are actually right?

This is the honest answer, not a marketing one: **always check `get_summary()` before the model score.**

```python
summary = pipeline.get_summary()

if not summary["success"]:
    raise RuntimeError(summary["errors"])          # a module failed outright

if summary["validation_problems"]:
    print("Something's off:", summary["validation_problems"])

if summary["data_quality_warnings"]:
    print("Worth reviewing:", summary["data_quality_warnings"])
```

`run()` raises `PipelineExecutionError` by default if a module fails, so a broken run won't quietly return something that looks like a result. Beyond that, `validate_results()` re-derives a handful of facts independently (does train+test add up to the full dataset, is there any row overlap between them, does the reported best model actually have the best score) instead of trusting what each module reported about itself — because "the pipeline didn't crash" and "the numbers are correct" are not the same guarantee.

---

## Domain packs

The core framework doesn't know or care what your columns mean — encoding falls back to automatic one-hot for anything it doesn't have an explicit mapping for, and no feature engineering is domain-specific unless you ask for it.

The oncology dataset this framework was originally built against is available as an opt-in domain pack:

```python
from ml_framework.config.config import FrameworkConfig

config = FrameworkConfig.from_domain("medical")  # known column mappings + engineered features for the reference dataset
pipeline = MLPipelineV2(config)
pipeline.run("ml_framework/data/cancer issue.csv", target_column="Recurrence")
```

Without `from_domain(...)`, `FrameworkConfig()` is domain-neutral — no column names, no medical vocabulary, no assumptions baked in. See `ml_framework/domain/medical/` for what the pack actually adds (encoding mappings, oncology-specific engineered features, clinical reporting).

---

## Key components

<details>
<summary><strong>Outlier treatment</strong></summary>

```python
from ml_framework.preprocessing.outlier_treatment import (
    OutlierTreatmentConfig, OutlierTreatmentSystem
)
# Strategies: cap | winsorize | transform | impute | scale | remove
config = OutlierTreatmentConfig(method="winsorize")
system = OutlierTreatmentSystem(config)
df_clean = system.apply(df, outliers_dict)
```
</details>

<details>
<summary><strong>Diagnostic & causal analysis</strong></summary>

```python
from ml_framework.analysis.diagnostic_analysis import diagnostic_analysis, causal_analysis

# Pearson |r| (numeric) + Cramér's V (categorical)
diag_df = diagnostic_analysis(df, target_col="target", top_n=10)

# Cohen's d effect sizes + statistical significance
causal_df = causal_analysis(df, target_col="target",
                            treatment_cols=["your_treatment_column"])
```
</details>

<details>
<summary><strong>Model Card (Google 2019 standard)</strong></summary>

```python
from ml_framework.modeling.model_card import generate_model_card, save_model_card

card = generate_model_card(model, X_train, X_test, y_train, y_test,
                           model_name="Your Model")
save_model_card(card, path="./reports/", filename="model_card")
# → reports/model_card.json  +  reports/model_card.md
```
</details>

<details>
<summary><strong>SHAP explainability</strong></summary>

```python
from ml_framework.interpretability.model_explainability import interpret_model_with_shap
interpret_model_with_shap(model, X_test, max_display=15, plot_type="summary")
```
</details>

<details>
<summary><strong>Data drift monitoring</strong></summary>

```python
from ml_framework.analysis.data_drift import detect_data_drift
drift_report = detect_data_drift(df_train, df_production)
# PSI < 0.10 : stable | 0.10-0.20 : warning | > 0.20 : retrain required
```
</details>

---

## Configuration

```python
from ml_framework.config.config import FrameworkConfig

config = FrameworkConfig()
config.model.test_size                     = 0.20
config.model.cv_folds                      = 5
config.model.models_to_test                = ["rf", "gb", "lr"]
config.model.perform_hyperparameter_tuning = True
config.data.missing_strategy               = "mice"   # mice | knn | miss_forest | simple
config.data.outlier_method                 = "iqr"    # iqr | zscore | modified_zscore | IsolationForest

pipeline = MLPipelineV2(config)
pipeline.run("your_data.csv", target_column="target")
```

---

## Project structure

<details>
<summary>Show full structure</summary>

```
dataAnalytic/
├── ml_framework/
│   ├── config/             # FrameworkConfig, ModelConfig, DataConfig, domain profiles
│   ├── data/                # Reference dataset (cancer issue.csv)
│   ├── domain/
│   │   └── medical/          # Optional oncology domain pack (opt-in, not a default)
│   ├── services/            # Data loading, decision support, documentation
│   ├── analysis/            # EDA, profiling, correlation, class imbalance, drift, diagnostics
│   ├── diagnostic/          # Class imbalance + leakage/data diagnostic checks
│   ├── insight/             # Automated business-insight pattern detection
│   ├── preprocessing/       # Cleaning, imputation, outliers, encoding, normalization
│   ├── features/            # Feature engineering, statistical + PCA selection
│   ├── strategies/          # Normalization decision logic, scoring strategy
│   ├── modeling/            # Models, training, ensembles, Model Card
│   ├── evaluation/          # Metrics, fairness, clinical report, overfitting check
│   ├── interpretability/    # SHAP, LIME, PDP, clinical risk profiles
│   ├── visualization/       # Plotting for analysis/evaluation/features/interpretability/monitoring
│   ├── monitoring/          # Production performance tracking
│   ├── optimization/        # Bayesian hyperparameter optimization
│   ├── core/                # Dataset split, export/import
│   ├── orchestration/       # v1 (pipeline.py) + v2/ (adaptive DAG orchestrator)
│   ├── tests/                # Integration tests
│   └── utils/                 # Metrics, display, plot helpers
├── notebook/
│   ├── test_recur.ipynb        # Full reference walkthrough on the oncology dataset
│   └── test_orchestrator.ipynb # v1 vs v2 orchestrator comparison
├── docs/
│   └── orchestration_v2_architecture.md
├── artifacts/ , reports/       # Generated at runtime (saved models, model cards, logs) — not source
└── README.md
```
</details>

---

## Installation

```bash
git clone <repo-url>
cd dataAnalytic
pip install pandas numpy scikit-learn matplotlib seaborn shap lime scipy statsmodels joblib xgboost lightgbm optuna
```

| Package | Used for |
|---------|----------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Models, preprocessing, evaluation |
| `matplotlib`, `seaborn` | Visualizations |
| `shap`, `lime` | Model explainability |
| `scipy`, `statsmodels` | Statistical tests |
| `joblib` | Model persistence |
| `xgboost`, `lightgbm` | Gradient boosting *(optional — the framework works without them, with a smaller model list)* |
| `optuna` | Bayesian hyperparameter tuning *(optional)* |

---

## Design principles

| Principle | How it shows up here |
|-----------|-----------------------|
| Zero data leakage | Train/test split happens before any transformer is fit |
| Domain-neutral core | No column names or domain vocabulary in default config; domain knowledge is opt-in |
| Adaptive, not fixed | v2's Decision Engine includes/skips steps based on what the data's own diagnostics found |
| Typed state | `PipelineContext` dataclass — no raw `dict` passed between steps |
| Fail loud, not quiet | A failed module raises by default instead of silently returning a partial result |
| Validate the result, not just the run | `get_summary()["validation_problems"]` re-derives basic facts instead of trusting each module's self-report |
| Full traceability | Execution times, errors, and every intermediate artifact are on `PipelineContext` |

---

## License

Educational and research use. Not intended for clinical, diagnostic, or commercial deployment.
