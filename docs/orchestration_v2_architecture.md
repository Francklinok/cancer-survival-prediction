# Pipeline Orchestration — Architecture Guide

**Status:** implemented and tested. This describes the orchestrator as it actually runs today, not a proposal.

If you're new to this codebase, read this document to understand *why* the orchestrator is built this way — what problem it solves, how the pieces fit together, and where to look when something needs to change.

---

## 1. The problem this solves

Before this design existed, the pipeline (`ml_framework/orchestration/pipeline.py`, still kept around as `MLPipeline` — see [Section 7](#7-v1-vs-v2-why-both-still-exist)) ran 12 fixed steps in a fixed order, every time, on every dataset:

```
ingest → profile → clean → eda → missing → outliers → encode → normalize → features → train → evaluate → persist
```

That's fine as a first version, but it has a real limitation: the pipeline can't adapt to the data. A dataset with zero missing values still pays for the `missing` step's setup cost. A dataset with no outliers still runs the outlier treatment. There was no clean way to ask "does this dataset actually need this step?" and act on the answer — each step had, at best, a hardcoded `skip_if` check buried inside its own class.

The v2 orchestrator (`ml_framework/orchestration/v2/`) fixes this by splitting "what should run" from "how it runs." A **Decision Engine** looks at what the data's diagnostics actually found and decides which modules belong in this run. A separate **Execution Engine** just executes whatever plan it's handed — it has no opinion about the data at all.

## 2. The four layers, in plain terms

Picture the request "run the pipeline on this CSV, target column Recurrence" moving through four stages:

```
                    MODULE REGISTRY
     (a catalog: what modules exist, what each one needs
      as input, what it produces, whether it can run in parallel)
                          │
                          ▼
   1. ANALYSIS ENGINE — runs diagnostic modules (missing-value scan,
      outlier detection, class balance, leakage check...) and translates
      each one's raw output into a common shape: "is this needed, and why?"
                          │  → a list of Recommendations
                          ▼
   2. DECISION ENGINE — takes those recommendations and decides which
      modules actually belong in this run. A dataset with no missing
      values simply doesn't get a "missing" step in its plan.
                          │  → an ExecutionPlan (ordered list of modules)
                          ▼
   3. DAG BUILDER — turns that plan into a dependency graph, using each
      module's declared inputs/outputs. If a module needs something no
      earlier module produces, this is where you find out — before
      anything runs, not halfway through a 10-minute training step.
                          │  → an ExecutionDAG
                          ▼
   4. EXECUTION ENGINE — walks the graph and actually runs each module,
      in dependency order, in parallel where the graph allows it. It
      never second-guesses the plan — by the time it gets a DAG, "what
      to run" has already been decided.
```

Two things worth calling out because they're easy to miss:

- **The Decision Engine never touches a DataFrame.** It only sees `Recommendation` objects and a `DatasetProfile`. If you're debugging "why did module X run/not run," you look at the Decision Engine's rules and the Recommendation that triggered them — not at pandas code.
- **The Execution Engine never decides anything.** If a module ran that shouldn't have, the bug is upstream, in the Decision Engine or the plan it produced — not in the Execution Engine itself.

## 3. Where recommendations come from: the Adapter pattern

Most of the framework's diagnostic functions (`diagnose_class_imbalance`, `leakage_exploration`, `suggest_normalization_strategy`, ...) were written long before this orchestrator existed, and they all return different shapes — a dict here, a DataFrame there, free-text recommendations in a list. Rewriting eighteen already-validated functions just so they'd return a uniform object wasn't worth the risk.

Instead, each one gets a small **adapter** — a pure function that reads the function's existing output and translates it into the one shared shape the Decision Engine understands:

```python
# specs/class_imbalance_adapter.py
def adapt(raw_output: dict) -> Recommendation:
    # raw_output is exactly what diagnose_class_imbalance() already returns —
    # nothing here re-analyzes the data, it just relabels the verdict.
    severity = raw_output["severity"]
    return Recommendation(
        topic="balancing",
        required=severity not in ("balanced", "mild"),
        strategy="SMOTE" if severity in ("severe", "extreme") else "class_weight",
        reason=severity,
        confidence=1.0,
        source_module="diagnose_class_imbalance",
        raw=raw_output,  # nothing is lost — you can always trace back to it
    )
```

This keeps the business logic completely untouched. If you ever need to check what a module *actually* found, `Recommendation.raw` still has it.

## 4. The building blocks

### Module Registry (`module_registry.py`)

The catalog. Every module — `ingest`, `clean`, `train`, and so on — is described once by a `ModuleSpec`:

```python
@dataclass(frozen=True)
class ModuleSpec:
    name: str                 # "train"
    version: str               # "1.0.0"
    capabilities: set[str]     # {"tabular"}
    inputs: set[str]           # what this module needs already available
    outputs: set[str]          # what it produces
    invoke: Callable            # the actual function that does the work
    cost_hint: str              # "low" | "medium" | "high" — a rough estimate, not a measurement
    parallelizable: bool
```

`inputs`/`outputs` matter more than they look — the DAG Builder relies on them being accurate. A recent example: `train` originally only declared `df_work` as required, but it actually needs the real train/test split that `normalize` produces. Because that dependency wasn't declared, a plan that skipped `normalize` could still "succeed" — `train` would silently fall back to re-splitting the data itself, on a smaller, already-split subset, producing numbers that looked fine but weren't measuring what they claimed to. Declaring the real inputs (`X_train`, `X_test`, `y_train`, `y_test`) means the DAG Builder now catches this at construction time, with a clear error, before a single model gets trained.

### Analysis Engine

Runs every module flagged `produces_recommendation=True`, applies its adapter, and hands the Decision Engine a `List[Recommendation]` plus a `DatasetProfile` (row/column counts, problem type). It doesn't make any decisions itself — it just gathers evidence.

### Decision Engine (`decision_engine.py`, `planning_rules.py`)

Turns recommendations into an `ExecutionPlan`. Each decision is its own small `PlanningRule` class:

- `AlwaysIncludeRule` — modules that always run (most of them, today — `ingest`, `clean`, `train`, etc.)
- `SkipIfNotRequiredRule` — include a module only if its `Recommendation.required` is `True` (used for `missing`, and optionally `outliers`/`normalize` when adaptive skipping is turned on)

Adding a new rule means adding a new class, not editing a growing `if/elif` chain.

### DAG Builder (`dag_builder.py`)

Takes the plan and the registry's declared `inputs`/`outputs`, and builds a dependency graph. Two things it catches before any code runs:

- **Missing input** — a module needs something no included module produces, and it's not supplied upfront either.
- **Cycle** — a module depends on something that can only come from a module running after it.

Both raise immediately, with a message naming the module and the missing/circular input, instead of failing confusingly deep into execution.

### Execution Engine (`execution_engine.py`)

Runs the DAG: topological order, modules with no dependency on each other run in parallel, retries on failure per the module's own settings, and — this matters — **a failed module stops the pipeline**. It doesn't quietly continue with partial data.

## 5. Trusting the result: what happens when something goes wrong

A pipeline that always "succeeds" isn't actually more trustworthy than one that fails loudly — it just moves the risk from "an error you see" to "a wrong number you don't." Two things exist specifically to close that gap:

**`run()` raises by default.** If any module fails, `MLPipelineV2.run(...)` raises `PipelineExecutionError` — you don't have to remember to check `step_errors` yourself. If you genuinely want the old best-effort behavior (record the failure, return a partial result), pass `raise_on_error=False` explicitly.

**`get_summary()` tells you what to double-check, not just what happened.** Beyond the usual `best_model`/`final_metrics`, it includes:

- `success` — `False` if anything failed (only reachable with `raise_on_error=False`)
- `validation_problems` — independent post-hoc sanity checks: do train + test row counts add up to the original dataset, is there any row-index overlap between train and test, does the reported `best_model` actually have the best score, was a model actually fitted. These checks don't trust any single module's own bookkeeping — they re-derive the answer and compare.
- `data_quality_warnings` — dataset-level risk signals the pipeline already computed but that don't block execution: severe class imbalance with no rebalancing applied, features flagged as likely target leakage.
- `analysis_warnings` — secondary analyses (SHAP, fairness audit, overfitting check, ...) that failed without stopping the run, listed instead of silently disappearing into a log line.

If you're running this on a dataset the framework hasn't seen before, `get_summary()` is the first thing to read — not the model score.

## 6. Worked example: how a dataset with no missing values gets a shorter plan

1. `eda` runs `diagnose_class_imbalance`, `leakage_exploration`, and a missing-value scan on the loaded data.
2. The missing-value scan finds 0 NaNs. Its adapter produces `Recommendation(topic="missing_values", required=False, reason="no NaN found")`.
3. The Decision Engine's `SkipIfNotRequiredRule("missing", topic="missing_values")` reads that recommendation and votes to exclude `missing` from the plan.
4. The DAG Builder never sees a `missing` node — nothing downstream depends on it being present, since `clean`'s output already satisfies whatever `outliers` needs next.
5. The Execution Engine runs the shorter plan. No code anywhere says `if column_has_no_nan: skip missing_step` — the data itself produced that outcome through the recommendation it generated.

## 7. v1 vs v2 — why both still exist

`ml_framework/orchestration/pipeline.py` (the original, still exposed as `MLPipeline`) is not dead code — it's the reference implementation v2 is validated against. `ml_framework/orchestration/v2/tests/test_phase3_non_regression.py` runs both engines on the real reference dataset with the same config and asserts their metrics match exactly. That test is the actual proof that v2 behaves correctly, not just that it runs without crashing.

**For new code, use v2** (`ml_framework.orchestration.v2.facade.MLPipelineV2`) — it has the adaptive planning, the DAG-level validation, and the reliability guarantees described in Section 5, none of which exist in v1. v1 stays in place as the non-regression baseline and isn't going away casually; removing it would mean losing the one automated proof that v2 is correct.

## 8. Extending this — adding a new module

1. Write the business logic as a plain function, wherever it belongs in the framework (e.g. `ml_framework/preprocessing/your_thing.py`). Don't touch the orchestrator to do this.
2. If the module should influence planning decisions (like `diagnose_class_imbalance` does), write a small adapter that maps its output to a `Recommendation`.
3. Declare a `ModuleSpec` in `ml_framework/orchestration/v2/specs/your_thing.py`: name, real `inputs`/`outputs`, and the `invoke` function that calls your business logic.
4. Register it. Add a `PlanningRule` if it should be conditionally included; otherwise `AlwaysIncludeRule` is enough.
5. Write a unit test for the adapter using a fixed, fake `raw_output` — you don't need a real dataset to test that a `Recommendation` gets built correctly.

That's the whole extension path — nothing above requires touching `execution_engine.py` or `dag_builder.py`, which stay generic on purpose.

## 9. What's deliberately not built

A few things were considered and left out, on purpose, because they'd add real complexity for a problem this project doesn't have yet:

- **A distributed event bus** (Kafka/Redis-style) — this runs as a single Python process executing one pipeline at a time. The in-process `EventBus` gives the same publish/subscribe interface; only the implementation would need to change if that ever stops being true.
- **Plugin auto-discovery** — every module lives in this repo, written by the same team. An explicit registry is more predictable than scanning packages for modules to auto-import, and predictable matters more than convenient here.
- **Multi-machine distributed execution** — the `ExecutionEngine` interface doesn't assume a single process, but the only implementation today is a local thread pool, because that's what the actual workload needs.
- **NLP/vision/time-series modules** — `DatasetProfile.problem_type` and `ModuleSpec.capabilities` exist as extension points, but no non-tabular module has been written. The hooks are there; the modules aren't, because this framework's only real dataset today is tabular.

None of this is a promise to build these later — it's a note that the design doesn't block them if the need ever shows up.
