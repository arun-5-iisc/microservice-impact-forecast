"""
Hyperparameter tuning for proactive deployment impact forecasting models.

Tunes RandomForest and LightGBM on both tasks using Optuna (Bayesian search)
with GroupKFold to prevent label leakage across the 4 T-snapshots per experiment.

Optimises:
  Classification — macro-F1 (maximise)
  Regression     — MAE (minimise)

Outputs:
  models/   — clf_rf_tuned.joblib, clf_lgbm_tuned.joblib,
               reg_rf_tuned.joblib, reg_lgbm_tuned.joblib
  results/  — best_params.json, tuning_report.csv

Usage:
  python scripts/tune_model.py
  python scripts/tune_model.py --data dataset/processed/training_dataset_fe.csv
  python scripts/tune_model.py --model rf          # RF only
  python scripts/tune_model.py --model lgbm        # LightGBM only
  python scripts/tune_model.py --trials 100        # more trials (default 50)
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
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from train_model import (
    _build_preprocessor,
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


def _refit_and_save(estimator_cls, best_params, X, y, numeric_features, path):
    pipe = Pipeline([
        ("pre", _build_preprocessor(numeric_features)),
        ("model", estimator_cls(**best_params)),
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, path)
    logger.info("  Saved → %s", path)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: Path, models: list[str], n_trials: int):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset: %d rows, %d experiments", len(df), df["experiment_id"].nunique())

    numeric_features = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in df.columns]
    all_features     = numeric_features + CATEGORICAL_FEATURES
    logger.info("Feature set: %d numeric + %d categorical", len(numeric_features), len(CATEGORICAL_FEATURES))

    X      = df[all_features]
    groups = df["experiment_id"].values

    # ── Classification setup ──────────────────────────────────────────────────
    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_ORDER)
    y_cls = le.transform(df["severity"])

    # ── Regression setup ──────────────────────────────────────────────────────
    y_reg = df["blast_radius_final"].values.astype(float)

    best_params_all = {}
    report_rows     = []

    # ── Baselines from train_model.py results (for comparison) ───────────────
    # These are the FE-dataset default hyperparameter results to compare against
    BASELINE_CLF = {"RF": 0.690, "LightGBM": 0.684}   # macro-F1
    BASELINE_REG = {"RF": 1.549, "LightGBM": 1.642}   # MAE

    # ── RandomForest ─────────────────────────────────────────────────────────
    if "rf" in models:
        logger.info("\n%s\nRandomForest Tuning\n%s", "="*55, "="*55)

        # Classification
        val, params, _ = _tune(
            "RF-clf", _rf_clf_objective, "maximize", n_trials,
            X, y_cls, groups, numeric_features,
        )
        clf_params = {**params, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
        _refit_and_save(RandomForestClassifier, clf_params, X, y_cls, numeric_features,
                        MODELS_DIR / "clf_rf_tuned.joblib")
        best_params_all["rf_clf"] = clf_params
        base = BASELINE_CLF["RF"]
        report_rows.append({
            "model": "RF", "task": "classification", "metric": "macro_f1",
            "baseline": round(base, 4), "tuned": round(val, 4),
            "improvement_pct": round(100 * (val - base) / base, 1), "n_trials": n_trials,
        })
        logger.info("  RF clf → baseline macro-F1=%.4f  tuned=%.4f  Δ=+%.1f%%",
                    base, val, 100 * (val - base) / base)

        # Regression
        val, params, _ = _tune(
            "RF-reg", _rf_reg_objective, "minimize", n_trials,
            X, y_reg, groups, numeric_features,
        )
        reg_params = {**params, "random_state": 42, "n_jobs": -1}
        _refit_and_save(RandomForestRegressor, reg_params, X, y_reg, numeric_features,
                        MODELS_DIR / "reg_rf_tuned.joblib")
        best_params_all["rf_reg"] = reg_params
        base = BASELINE_REG["RF"]
        report_rows.append({
            "model": "RF", "task": "regression", "metric": "mae",
            "baseline": round(base, 4), "tuned": round(val, 4),
            "improvement_pct": round(100 * (base - val) / base, 1), "n_trials": n_trials,
        })
        logger.info("  RF reg → baseline MAE=%.4f  tuned=%.4f  Δ=-%.1f%%",
                    base, val, 100 * (base - val) / base)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    if "lgbm" in models:
        logger.info("\n%s\nLightGBM Tuning\n%s", "="*55, "="*55)

        # Classification
        val, params, _ = _tune(
            "LGBM-clf", _lgbm_clf_objective, "maximize", n_trials,
            X, y_cls, groups, numeric_features,
        )
        clf_params = {**params, "class_weight": "balanced", "random_state": 42, "verbose": -1}
        _refit_and_save(LGBMClassifier, clf_params, X, y_cls, numeric_features,
                        MODELS_DIR / "clf_lgbm_tuned.joblib")
        best_params_all["lgbm_clf"] = clf_params
        base = BASELINE_CLF["LightGBM"]
        report_rows.append({
            "model": "LightGBM", "task": "classification", "metric": "macro_f1",
            "baseline": round(base, 4), "tuned": round(val, 4),
            "improvement_pct": round(100 * (val - base) / base, 1), "n_trials": n_trials,
        })
        logger.info("  LGBM clf → baseline macro-F1=%.4f  tuned=%.4f  Δ=+%.1f%%",
                    base, val, 100 * (val - base) / base)

        # Regression
        val, params, _ = _tune(
            "LGBM-reg", _lgbm_reg_objective, "minimize", n_trials,
            X, y_reg, groups, numeric_features,
        )
        reg_params = {**params, "random_state": 42, "verbose": -1}
        _refit_and_save(LGBMRegressor, reg_params, X, y_reg, numeric_features,
                        MODELS_DIR / "reg_lgbm_tuned.joblib")
        best_params_all["lgbm_reg"] = reg_params
        base = BASELINE_REG["LightGBM"]
        report_rows.append({
            "model": "LightGBM", "task": "regression", "metric": "mae",
            "baseline": round(base, 4), "tuned": round(val, 4),
            "improvement_pct": round(100 * (base - val) / base, 1), "n_trials": n_trials,
        })
        logger.info("  LGBM reg → baseline MAE=%.4f  tuned=%.4f  Δ=-%.1f%%",
                    base, val, 100 * (base - val) / base)

    # ── Save outputs ──────────────────────────────────────────────────────────
    params_path = RESULTS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params_all, f, indent=2)
    logger.info("Best params → %s", params_path)

    report_df = pd.DataFrame(report_rows)
    report_path = RESULTS_DIR / "tuning_report.csv"
    report_df.to_csv(report_path, index=False)

    logger.info("\nTuning report:\n%s", report_df.to_string(index=False))
    logger.info("Done. Models saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--data",   type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model",  choices=["rf", "lgbm", "all"], default="all")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    models = ["rf", "lgbm"] if args.model == "all" else [args.model]
    main(args.data, models, args.trials)
