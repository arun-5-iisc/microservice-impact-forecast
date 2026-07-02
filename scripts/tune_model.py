"""
Hyperparameter tuning for proactive deployment impact forecasting models.

Tunes RandomForest, XGBoost, and LightGBM on both tasks using Optuna (Bayesian
search) with GroupKFold to prevent label leakage across the 4 T-snapshots per
experiment.

Optimises:
  Classification — macro-F1 (maximise)
  Regression     — MAE (minimise)

Experimental flow (two-stage isolation design):
  Stage 1: --data training_dataset.csv    --stage baseline_tuned
           Tune all models on original features → pick best algorithm
  Stage 2: --data training_dataset_fe.csv --stage fe_tuned   --model <winner>
           Tune winner on FE features → isolates FE contribution

After tuning, runs a full OOF evaluation (accuracy, per-class F1, MAE, RMSE)
and writes the comparison table to results/<stage>/final_results.csv.

Outputs (all under results/<stage>/ and models/<stage>/):
  models/<stage>/   — clf_rf_tuned.joblib, clf_xgb_tuned.joblib, clf_lgbm_tuned.joblib
                       reg_rf_tuned.joblib, reg_xgb_tuned.joblib, reg_lgbm_tuned.joblib
  results/<stage>/  — best_params.json, tuning_report.csv,
                       final_results.csv, accuracy_vs_time_tuned.csv

Usage:
  # Stage 1: tune all models on baseline dataset
  python scripts/tune_model.py --data dataset/processed/training_dataset.csv --stage baseline_tuned

  # Stage 2: tune winner on FE dataset
  python scripts/tune_model.py --data dataset/processed/training_dataset_fe.csv --stage fe_tuned --model lgbm

  # Re-run evaluation only (load saved best_params.json)
  python scripts/tune_model.py --stage baseline_tuned --eval-only

  # Explicit trial count
  python scripts/tune_model.py --stage baseline_tuned --trials 100
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from train_model import (
    _build_preprocessor,
    _evaluate_classifier,
    _evaluate_regressor,
    CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    FE_NUMERIC_FEATURES,
    SEVERITY_ORDER,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "processed" / "training_dataset_fe.csv"
MODELS_DIR   = REPO_ROOT / "models"
RESULTS_DIR  = REPO_ROOT / "results"

N_FOLDS   = 5
N_TRIALS  = 50


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objectives
# ─────────────────────────────────────────────────────────────────────────────

def _rf_clf_objective(trial, X, y, groups, numeric_features):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":         trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
    }
    estimator = RandomForestClassifier(**params)
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        scores.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
    return np.mean(scores)


def _lgbm_clf_objective(trial, X, y, groups, numeric_features):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight":      "balanced",
        "random_state":      42,
        "verbose":           -1,
    }
    estimator = LGBMClassifier(**params)
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        scores.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
    return np.mean(scores)


def _rf_reg_objective(trial, X, y, groups, numeric_features):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":         trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "random_state":      42,
        "n_jobs":            -1,
    }
    estimator = RandomForestRegressor(**params)
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        scores.append(mean_absolute_error(y[test_idx], y_pred))
    return np.mean(scores)


def _lgbm_reg_objective(trial, X, y, groups, numeric_features):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state":      42,
        "verbose":           -1,
    }
    estimator = LGBMRegressor(**params)
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        scores.append(mean_absolute_error(y[test_idx], y_pred))
    return np.mean(scores)


def _xgb_clf_objective(trial, X, y, groups, numeric_features):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    estimator = XGBClassifier(**params)
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        sw = compute_sample_weight("balanced", y[train_idx])
        pipe.fit(X.iloc[train_idx], y[train_idx], model__sample_weight=sw)
        y_pred = pipe.predict(X.iloc[test_idx])
        scores.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
    return np.mean(scores)


def _xgb_reg_objective(trial, X, y, groups, numeric_features):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    estimator = XGBRegressor(**params)
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        scores.append(mean_absolute_error(y[test_idx], y_pred))
    return np.mean(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Generic tune + refit helper
# ─────────────────────────────────────────────────────────────────────────────

def _tune(name: str, objective_fn, direction: str, n_trials: int,
          X, y, groups, numeric_features):
    logger.info("Tuning %s (%s, %d trials)…", name, direction, n_trials)
    t0 = time.time()

    def _wrapped(trial):
        return objective_fn(trial, X, y, groups, numeric_features)

    study = optuna.create_study(direction=direction)
    study.optimize(
        _wrapped,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_log_callback(name, direction)],
    )
    elapsed = time.time() - t0
    logger.info(
        "  %s done: best=%.4f  params=%s  (%.0fs)",
        name, study.best_value, study.best_params, elapsed,
    )
    return study.best_value, study.best_params, study


def _log_callback(name, direction):
    """Print a one-liner every 10 trials."""
    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == 0:
            logger.info(
                "  [%s] trial %d: value=%.4f  best=%.4f",
                name, trial.number, trial.value, study.best_value,
            )
    return callback


def _save_study_plots(study: optuna.Study, name: str, plots_dir: Path) -> None:
    """Save optimization history and hyperparameter importance plots as PNG.

    Requires kaleido: pip install kaleido
    """
    try:
        import optuna.visualization as ov
        plots_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace("-", "_")

        fig = ov.plot_optimization_history(study)
        fig.update_layout(title=f"{name} — Optuna Convergence")
        fig.write_image(str(plots_dir / f"convergence_{safe_name}.png"))

        fig = ov.plot_param_importances(study)
        fig.update_layout(title=f"{name} — Hyperparameter Importance")
        fig.write_image(str(plots_dir / f"param_importance_{safe_name}.png"))

        logger.info("  Plots saved → %s/{convergence,param_importance}_%s.png",
                    plots_dir, safe_name)
    except ImportError:
        logger.warning("  kaleido not installed — skipping plots. "
                       "Install with: pip install kaleido")
    except Exception as exc:
        logger.warning("  Plot export failed (%s): %s", name, exc)


def _refit_and_save(estimator_cls, best_params, X, y, numeric_features, path,
                    fit_params: dict | None = None):
    pipe = Pipeline([
        ("pre", _build_preprocessor(numeric_features)),
        ("model", estimator_cls(**best_params)),
    ])
    pipe.fit(X, y, **(fit_params or {}))
    joblib.dump(pipe, path)
    logger.info("  Saved → %s", path)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Full OOF evaluation for tuned models
# ─────────────────────────────────────────────────────────────────────────────

PRIOR_COLS = ["stage", "model", "accuracy", "macro_f1", "Low_f1",
              "Medium_f1", "High_f1", "blast_mae", "blast_rmse"]

# Fallback numbers used only when Stage 0 / Stage 1 CSVs are absent.
# These will be replaced by actual CSV results at runtime once those stages run.
_PRIOR_RESULTS_FALLBACK = [
    # stage, model, accuracy, macro_f1, Low_f1, Medium_f1, High_f1, blast_mae, blast_rmse
    ("baseline", "RandomForest", 0.697,  0.610, 0.758, 0.266, 0.807, 1.639, 2.016),
    ("baseline", "XGBoost",      0.722,  0.634, 0.757, 0.329, 0.814, 1.717, 2.116),
    ("baseline", "LightGBM",     0.758,  0.684, 0.781, 0.414, 0.857, 1.642, 2.055),
    ("fe",       "RandomForest", 0.753,  0.690, 0.755, 0.432, 0.885, 1.549, 1.946),
    ("fe",       "XGBoost",      0.736,  0.661, 0.737, 0.403, 0.842, 1.662, 2.108),
    ("fe",       "LightGBM",     0.744,  0.669, 0.758, 0.392, 0.857, 1.613, 2.017),
]


def _load_prior_results() -> list[tuple]:
    """Load Stage 0 and Stage 1 results from CSV files.

    Reads results/<stage>/classification_report.csv and regression_report.csv
    for both 'baseline' and 'fe' stages. Falls back to _PRIOR_RESULTS_FALLBACK
    when the files do not yet exist (i.e. Stage 0/1 have not run yet).
    """
    rows = []
    for stage_key in ("baseline", "fe"):
        clf_path = RESULTS_DIR / stage_key / "classification_report.csv"
        reg_path = RESULTS_DIR / stage_key / "regression_report.csv"
        if not clf_path.exists() or not reg_path.exists():
            continue
        try:
            clf_df = pd.read_csv(clf_path)
            reg_df = pd.read_csv(reg_path)
        except Exception as exc:
            logger.warning("Could not read prior results for stage '%s': %s", stage_key, exc)
            continue

        for _, clf_row in clf_df.iterrows():
            model = clf_row["model"]
            if model == "RuleBaseline":
                continue
            reg_match = reg_df[reg_df["model"] == model]
            if reg_match.empty:
                continue
            reg_row = reg_match.iloc[0]
            rows.append((
                stage_key,
                model,
                float(clf_row["accuracy"]),
                float(clf_row["macro_f1"]),
                float(clf_row["Low_f1"]),
                float(clf_row["Medium_f1"]),
                float(clf_row["High_f1"]),
                float(reg_row["mae_mean"]),
                float(reg_row["rmse_mean"]),
            ))

    if rows:
        logger.info("Loaded %d prior result rows from Stage 0/1 CSVs", len(rows))
        return rows

    logger.warning(
        "Stage 0/1 result CSVs not found — using fallback prior results. "
        "Run train_model.py --stage baseline and --stage fe first."
    )
    return list(_PRIOR_RESULTS_FALLBACK)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model dispatch tables and param finalisation
# ─────────────────────────────────────────────────────────────────────────────

_CLF_CLS = {"rf": RandomForestClassifier, "xgb": XGBClassifier, "lgbm": LGBMClassifier}
_REG_CLS = {"rf": RandomForestRegressor,  "xgb": XGBRegressor,  "lgbm": LGBMRegressor}
_CLF_OBJ = {"rf": _rf_clf_objective, "xgb": _xgb_clf_objective, "lgbm": _lgbm_clf_objective}
_REG_OBJ = {"rf": _rf_reg_objective, "xgb": _xgb_reg_objective, "lgbm": _lgbm_reg_objective}
_LABELS  = {"rf": "RandomForest", "xgb": "XGBoost", "lgbm": "LightGBM"}


def _finalize_clf_params(model: str, tuned: dict) -> dict:
    """Merge Optuna-found params with fixed non-tuned fields for each classifier."""
    base = dict(tuned)
    if model == "rf":
        base.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1})
    elif model == "xgb":
        base.update({"random_state": 42, "n_jobs": -1, "verbosity": 0})
    elif model == "lgbm":
        base.update({"class_weight": "balanced", "random_state": 42, "verbose": -1})
    return base


def _finalize_reg_params(model: str, tuned: dict) -> dict:
    """Merge Optuna-found params with fixed non-tuned fields for each regressor."""
    base = dict(tuned)
    if model == "rf":
        base.update({"random_state": 42, "n_jobs": -1})
    elif model == "xgb":
        base.update({"random_state": 42, "n_jobs": -1, "verbosity": 0})
    elif model == "lgbm":
        base.update({"random_state": 42, "verbose": -1})
    return base


def _get_clf_fit_params(model: str, y_cls: np.ndarray) -> dict | None:
    """Return Pipeline.fit() kwargs for class-imbalance handling (XGBoost only)."""
    if model == "xgb":
        return {"model__sample_weight": compute_sample_weight("balanced", y_cls)}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main — two independent pipelines: best clf model + best reg model
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: Path, stage: str, clf_model: str, reg_model: str,
         n_trials: int, eval_only: bool = False):
    models_dir  = MODELS_DIR  / stage
    results_dir = RESULTS_DIR / stage
    plots_dir   = results_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s  [stage=%s]", data_path, stage)
    logger.info("Classification winner: %s  |  Regression winner: %s",
                _LABELS[clf_model], _LABELS[reg_model])
    df = pd.read_csv(data_path)
    logger.info("Dataset: %d rows, %d experiments", len(df), df["experiment_id"].nunique())

    # ── Train / holdout split ─────────────────────────────────────────────────
    holdout_path = REPO_ROOT / "dataset" / "processed" / "holdout_ids.txt"
    if holdout_path.exists():
        holdout_ids = set(holdout_path.read_text().strip().splitlines())
        df_train    = df[~df["experiment_id"].isin(holdout_ids)].reset_index(drop=True)
        df_test     = df[ df["experiment_id"].isin(holdout_ids)].reset_index(drop=True)
        logger.info(
            "Holdout split: %d training experiments (%d rows)  +  "
            "%d holdout experiments (%d rows)",
            df_train["experiment_id"].nunique(), len(df_train),
            df_test["experiment_id"].nunique(), len(df_test),
        )
    else:
        logger.warning("holdout_ids.txt not found — using all data for tuning.")
        df_train = df.reset_index(drop=True)
        df_test  = None

    numeric_features = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in df_train.columns]
    all_features     = numeric_features + CATEGORICAL_FEATURES
    logger.info("Feature set: %d numeric + %d categorical", len(numeric_features), len(CATEGORICAL_FEATURES))

    X      = df_train[all_features]
    groups = df_train["experiment_id"].values

    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_ORDER)
    y_cls = le.transform(df_train["severity"])
    y_reg = df_train["blast_radius_final"].values.astype(float)

    best_params_all = {}
    report_rows     = []

    # Stage-aware prior results: compare tuning gain against the matching default-param stage
    prior_stage  = "baseline" if "baseline" in stage else "fe"
    all_prior    = _load_prior_results()
    prior_lookup = {
        row[1]: {"macro_f1": row[3], "mae": row[7]}
        for row in all_prior
        if row[0] == prior_stage
    }

    # ── Load existing params if eval-only ────────────────────────────────────
    params_path = results_dir / "best_params.json"
    if eval_only:
        if not params_path.exists():
            logger.error("--eval-only requires %s to exist. Run tuning first.", params_path)
            return
        with open(params_path) as f:
            best_params_all = json.load(f)
        logger.info("Loaded best params from %s", params_path)

    # ── Classification pipeline: tune clf_model ───────────────────────────────
    logger.info("\n%s\nClassification Pipeline — tuning %s\n%s",
                "="*55, _LABELS[clf_model], "="*55)
    clf_key = f"{clf_model}_clf"
    if not eval_only:
        val, params, study = _tune(
            f"{clf_model.upper()}-clf", _CLF_OBJ[clf_model], "maximize", n_trials,
            X, y_cls, groups, numeric_features,
        )
        _save_study_plots(study, f"{clf_model}-clf", plots_dir)
        clf_params = _finalize_clf_params(clf_model, params)
        _refit_and_save(
            _CLF_CLS[clf_model], clf_params, X, y_cls, numeric_features,
            models_dir / f"clf_{clf_model}_tuned.joblib",
            fit_params=_get_clf_fit_params(clf_model, y_cls),
        )
        best_params_all[clf_key] = clf_params
        base = prior_lookup.get(_LABELS[clf_model], {}).get("macro_f1", 0.0)
        report_rows.append({
            "model": _LABELS[clf_model], "task": "classification", "metric": "macro_f1",
            "baseline": round(base, 4), "tuned": round(val, 4),
            "improvement_pct": round(100 * (val - base) / max(base, 1e-9), 1),
            "n_trials": n_trials,
        })
        logger.info("  %s clf → baseline=%.4f  tuned=%.4f  Δ=+%.1f%%",
                    _LABELS[clf_model], base, val, 100 * (val - base) / max(base, 1e-9))

    # ── Regression pipeline: tune reg_model ───────────────────────────────────
    logger.info("\n%s\nRegression Pipeline — tuning %s\n%s",
                "="*55, _LABELS[reg_model], "="*55)
    reg_key = f"{reg_model}_reg"
    if not eval_only:
        val, params, study = _tune(
            f"{reg_model.upper()}-reg", _REG_OBJ[reg_model], "minimize", n_trials,
            X, y_reg, groups, numeric_features,
        )
        _save_study_plots(study, f"{reg_model}-reg", plots_dir)
        reg_params = _finalize_reg_params(reg_model, params)
        _refit_and_save(
            _REG_CLS[reg_model], reg_params, X, y_reg, numeric_features,
            models_dir / f"reg_{reg_model}_tuned.joblib",
        )
        best_params_all[reg_key] = reg_params
        base = prior_lookup.get(_LABELS[reg_model], {}).get("mae", 0.0)
        report_rows.append({
            "model": _LABELS[reg_model], "task": "regression", "metric": "mae",
            "baseline": round(base, 4), "tuned": round(val, 4),
            "improvement_pct": round(100 * (base - val) / max(base, 1e-9), 1),
            "n_trials": n_trials,
        })
        logger.info("  %s reg → baseline=%.4f  tuned=%.4f  Δ=-%.1f%%",
                    _LABELS[reg_model], base, val, 100 * (base - val) / max(base, 1e-9))

    # ── Save tuning report ────────────────────────────────────────────────────
    if not eval_only:
        with open(params_path, "w") as f:
            json.dump(best_params_all, f, indent=2)
        pd.DataFrame(report_rows).to_csv(results_dir / "tuning_report.csv", index=False)
        logger.info("Best params → %s", params_path)

    # ── Full OOF evaluation ───────────────────────────────────────────────────
    logger.info("\n%s\nFull OOF Evaluation\n%s", "="*55, "="*55)

    final_rows  = list(all_prior)
    all_t_rows  = []
    oof_records = []
    t_values    = df_train["time_into_fault_s"].values

    # Classification: evaluate tuned clf_model
    if clf_key in best_params_all:
        logger.info("  OOF eval — %s classifier (tuned)…", _LABELS[clf_model])
        _, clf_report, _, oof_preds, oof_true = _evaluate_classifier(
            _LABELS[clf_model],
            _CLF_CLS[clf_model](**best_params_all[clf_key]),
            X, y_cls, groups, le, numeric_features,
        )
        for T in [15, 30, 45, 60]:
            mask = t_values == T
            if mask.sum() == 0:
                continue
            acc = accuracy_score(oof_true[mask], oof_preds[mask])
            f1  = f1_score(oof_true[mask], oof_preds[mask], average="macro", zero_division=0)
            all_t_rows.append({
                "model": _LABELS[clf_model], "time_into_fault_s": T,
                "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
            })
            logger.info("    T=%2ds → accuracy=%.3f  macro-F1=%.3f", T, acc, f1)
        for pred, true in zip(oof_preds, oof_true):
            oof_records.append({
                "model":      _LABELS[clf_model],
                "true_label": le.classes_[true],
                "pred_label": le.classes_[pred],
            })
        # Stage 2 clf row: reg metrics are NaN (not this model's task)
        final_rows.append((
            stage, _LABELS[clf_model],
            round(clf_report["accuracy"], 4),
            round(clf_report["macro avg"]["f1-score"], 4),
            round(clf_report["Low"]["f1-score"], 4),
            round(clf_report["Medium"]["f1-score"], 4),
            round(clf_report["High"]["f1-score"], 4),
            np.nan, np.nan,
        ))
        logger.info(
            "  %s clf tuned → accuracy=%.3f  macro-F1=%.3f  [Low=%.3f  Med=%.3f  High=%.3f]",
            _LABELS[clf_model],
            clf_report["accuracy"], clf_report["macro avg"]["f1-score"],
            clf_report["Low"]["f1-score"], clf_report["Medium"]["f1-score"],
            clf_report["High"]["f1-score"],
        )

    # Regression: evaluate tuned reg_model
    if reg_key in best_params_all:
        logger.info("  OOF eval — %s regressor (tuned)…", _LABELS[reg_model])
        _, reg_metrics, _ = _evaluate_regressor(
            _LABELS[reg_model],
            _REG_CLS[reg_model](**best_params_all[reg_key]),
            X, y_reg, groups, numeric_features,
        )
        # Stage 2 reg row: clf metrics are NaN (not this model's task)
        final_rows.append((
            stage, _LABELS[reg_model],
            np.nan, np.nan, np.nan, np.nan, np.nan,
            round(reg_metrics["mae_mean"], 4),
            round(reg_metrics["rmse_mean"], 4),
        ))
        logger.info("  %s reg tuned → MAE=%.3f  RMSE=%.3f",
                    _LABELS[reg_model], reg_metrics["mae_mean"], reg_metrics["rmse_mean"])

    # ── Save results ──────────────────────────────────────────────────────────
    final_df = pd.DataFrame(final_rows, columns=PRIOR_COLS)
    final_df.to_csv(results_dir / "final_results.csv", index=False)
    logger.info("\nFull three-stage comparison:\n%s", final_df.to_string(index=False))

    if all_t_rows:
        pd.DataFrame(all_t_rows).to_csv(results_dir / "accuracy_vs_time_tuned.csv", index=False)

    if oof_records:
        pd.DataFrame(oof_records).to_csv(results_dir / "oof_predictions.csv", index=False)
        logger.info("OOF predictions → %s", results_dir / "oof_predictions.csv")

    # ── Final holdout evaluation (20 unseen experiments) ─────────────────────
    if df_test is not None and len(df_test) > 0:
        logger.info("\n%s\nFinal Holdout Evaluation (%d unseen experiments)\n%s",
                    "="*55, df_test["experiment_id"].nunique(), "="*55)

        X_test     = df_test[all_features]
        y_cls_test = le.transform(df_test["severity"])
        y_reg_test = df_test["blast_radius_final"].values.astype(float)

        # Best classifier on holdout
        clf_path = models_dir / f"clf_{clf_model}_tuned.joblib"
        if clf_path.exists():
            pipe   = joblib.load(clf_path)
            y_pred = pipe.predict(X_test)
            report = classification_report(
                y_cls_test, y_pred,
                labels=list(range(len(SEVERITY_ORDER))),
                target_names=SEVERITY_ORDER,
                output_dict=True,
                zero_division=0,
            )
            acc = accuracy_score(y_cls_test, y_pred)
            h_row = {"model": _LABELS[clf_model], "accuracy": round(acc, 4),
                     "macro_f1": round(report["macro avg"]["f1-score"], 4)}
            for cls in SEVERITY_ORDER:
                h_row[f"{cls}_f1"] = round(report[cls]["f1-score"], 4)
            pd.DataFrame([h_row]).to_csv(results_dir / "holdout_report.csv", index=False)
            logger.info(
                "  %s clf holdout → accuracy=%.3f  macro-F1=%.3f  "
                "[Low=%.3f  Med=%.3f  High=%.3f]",
                _LABELS[clf_model], acc,
                report["macro avg"]["f1-score"],
                report["Low"]["f1-score"], report["Medium"]["f1-score"],
                report["High"]["f1-score"],
            )

        # Best regressor on holdout
        reg_path = models_dir / f"reg_{reg_model}_tuned.joblib"
        if reg_path.exists():
            pipe   = joblib.load(reg_path)
            y_pred = pipe.predict(X_test)
            mae    = mean_absolute_error(y_reg_test, y_pred)
            rmse   = float(np.sqrt(np.mean((y_reg_test - y_pred) ** 2)))
            pd.DataFrame([{
                "model": _LABELS[reg_model], "mae": round(mae, 4), "rmse": round(rmse, 4),
            }]).to_csv(results_dir / "holdout_regression_report.csv", index=False)
            logger.info("  %s reg holdout → MAE=%.3f  RMSE=%.3f",
                        _LABELS[reg_model], mae, rmse)

    logger.info("\nDone. Models → %s/", models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--data",      type=Path, default=DEFAULT_DATA,
                        help="Training dataset CSV")
    parser.add_argument("--stage",     type=str,  default=None,
                        help="Output sub-directory (auto-detected from --data if omitted)")
    parser.add_argument("--clf-model", dest="clf_model", choices=["rf", "xgb", "lgbm"],
                        default="lgbm",
                        help="Best classifier from Stage 1 (default: lgbm)")
    parser.add_argument("--reg-model", dest="reg_model", choices=["rf", "xgb", "lgbm"],
                        default="rf",
                        help="Best regressor from Stage 1 (default: rf)")
    parser.add_argument("--trials",    type=int, default=N_TRIALS)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Optuna; load best_params.json and run full evaluation only")
    args = parser.parse_args()

    stage = args.stage or (
        "baseline_tuned" if "fe" not in args.data.stem else "fe_tuned"
    )
    main(args.data, stage, args.clf_model, args.reg_model, args.trials, args.eval_only)
