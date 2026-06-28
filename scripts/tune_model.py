"""
Hyperparameter tuning for proactive deployment impact forecasting models.

Tunes RandomForest and LightGBM on both tasks using Optuna (Bayesian search)
with GroupKFold to prevent label leakage across the 4 T-snapshots per experiment.

Optimises:
  Classification — macro-F1 (maximise)
  Regression     — MAE (minimise)

After tuning, runs a full OOF evaluation (accuracy, per-class F1, MAE, RMSE)
and writes the complete three-stage comparison table to final_results.csv.

Outputs:
  models/   — clf_rf_tuned.joblib, clf_lgbm_tuned.joblib,
               reg_rf_tuned.joblib, reg_lgbm_tuned.joblib
  results/  — best_params.json, tuning_report.csv,
               final_results.csv, accuracy_vs_time_tuned.csv

Usage:
  python scripts/tune_model.py                        # tune + evaluate
  python scripts/tune_model.py --eval-only            # skip Optuna, load best_params.json
  python scripts/tune_model.py --data <path>
  python scripts/tune_model.py --model rf             # RF only
  python scripts/tune_model.py --model lgbm           # LightGBM only
  python scripts/tune_model.py --trials 100
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
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor

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
# Full OOF evaluation for tuned models
# ─────────────────────────────────────────────────────────────────────────────

# Prior stage results (hardcoded from train_model.py outputs — frozen, reproducible)
PRIOR_RESULTS = [
    # stage, model, accuracy, macro_f1, Low_f1, Medium_f1, High_f1, blast_mae, blast_rmse
    ("baseline", "RandomForest", 0.697,  0.610, 0.758, 0.266, 0.807, 1.639, 2.016),
    ("baseline", "XGBoost",      0.722,  0.634, 0.757, 0.329, 0.814, 1.717, 2.116),
    ("baseline", "LightGBM",     0.758,  0.684, 0.781, 0.414, 0.857, 1.642, 2.055),
    ("fe",       "RandomForest", 0.753,  0.690, 0.755, 0.432, 0.885, 1.549, 1.946),
    ("fe",       "XGBoost",      0.736,  0.661, 0.737, 0.403, 0.842, 1.662, 2.108),
    ("fe",       "LightGBM",     0.744,  0.669, 0.758, 0.392, 0.857, 1.613, 2.017),
]

PRIOR_COLS = ["stage", "model", "accuracy", "macro_f1", "Low_f1",
              "Medium_f1", "High_f1", "blast_mae", "blast_rmse"]


def _full_eval_tuned(model_name: str, clf_cls, clf_params: dict,
                     reg_cls, reg_params: dict,
                     X, y_cls, y_reg, groups, le, numeric_features,
                     t_values: np.ndarray) -> dict:
    """
    Run full OOF evaluation (accuracy, per-class F1, MAE) for one tuned model.
    Reuses _evaluate_classifier / _evaluate_regressor from train_model.py.
    """
    logger.info("  Full OOF eval for %s (tuned)…", model_name)

    _, clf_report, _, oof_preds, oof_true = _evaluate_classifier(
        model_name, clf_cls(**clf_params), X, y_cls, groups, le, numeric_features
    )
    _, reg_metrics, _ = _evaluate_regressor(
        model_name, reg_cls(**reg_params), X, y_reg, groups, numeric_features
    )

    # Time-into-fault breakdown using OOF predictions
    t_rows = []
    for T in [15, 30, 45, 60]:
        mask = t_values == T
        if mask.sum() == 0:
            continue
        acc = accuracy_score(oof_true[mask], oof_preds[mask])
        f1  = f1_score(oof_true[mask], oof_preds[mask], average="macro", zero_division=0)
        t_rows.append({"model": model_name, "time_into_fault_s": T,
                        "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        logger.info("    T=%2ds → accuracy=%.3f  macro-F1=%.3f", T, acc, f1)

    return {
        "accuracy":   round(clf_report["accuracy"], 4),
        "macro_f1":   round(clf_report["macro avg"]["f1-score"], 4),
        "Low_f1":     round(clf_report["Low"]["f1-score"], 4),
        "Medium_f1":  round(clf_report["Medium"]["f1-score"], 4),
        "High_f1":    round(clf_report["High"]["f1-score"], 4),
        "blast_mae":  round(reg_metrics["mae_mean"], 4),
        "blast_rmse": round(reg_metrics["rmse_mean"], 4),
        "t_rows":     t_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: Path, models: list[str], n_trials: int, eval_only: bool = False):
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
    BASELINE_CLF = {"RF": 0.690, "LightGBM": 0.684}   # macro-F1 on FE dataset
    BASELINE_REG = {"RF": 1.549, "LightGBM": 1.642}   # MAE on FE dataset

    # ── Load existing params if eval-only ────────────────────────────────────
    params_path = RESULTS_DIR / "best_params.json"
    if eval_only:
        if not params_path.exists():
            logger.error("--eval-only requires %s to exist. Run tuning first.", params_path)
            return
        with open(params_path) as f:
            best_params_all = json.load(f)
        logger.info("Loaded best params from %s", params_path)

    # ── RandomForest ─────────────────────────────────────────────────────────
    if "rf" in models and not eval_only:
        logger.info("\n%s\nRandomForest Tuning\n%s", "="*55, "="*55)

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
        logger.info("  RF clf → baseline=%.4f  tuned=%.4f  Δ=+%.1f%%",
                    base, val, 100 * (val - base) / base)

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
        logger.info("  RF reg → baseline=%.4f  tuned=%.4f  Δ=-%.1f%%",
                    base, val, 100 * (base - val) / base)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    if "lgbm" in models and not eval_only:
        logger.info("\n%s\nLightGBM Tuning\n%s", "="*55, "="*55)

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
        logger.info("  LGBM clf → baseline=%.4f  tuned=%.4f  Δ=+%.1f%%",
                    base, val, 100 * (val - base) / base)

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
        logger.info("  LGBM reg → baseline=%.4f  tuned=%.4f  Δ=-%.1f%%",
                    base, val, 100 * (base - val) / base)

    # ── Save tuning params + report (skip in eval-only mode) ─────────────────
    if not eval_only:
        with open(params_path, "w") as f:
            json.dump(best_params_all, f, indent=2)
        logger.info("Best params → %s", params_path)

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(RESULTS_DIR / "tuning_report.csv", index=False)
        logger.info("\nTuning report:\n%s", report_df.to_string(index=False))

    # ── Full OOF evaluation of tuned models ───────────────────────────────────
    logger.info("\n%s\nFull Evaluation: Tuned Models\n%s", "="*55, "="*55)

    t_values   = df["time_into_fault_s"].values
    final_rows = list(PRIOR_RESULTS)   # start with frozen baseline + FE rows
    all_t_rows = []

    eval_map = {
        "rf":   (RandomForestClassifier, "rf_clf",   RandomForestRegressor, "rf_reg"),
        "lgbm": (LGBMClassifier,         "lgbm_clf", LGBMRegressor,         "lgbm_reg"),
    }
    model_labels = {"rf": "RandomForest", "lgbm": "LightGBM"}

    for m in models:
        clf_cls, clf_key, reg_cls, reg_key = eval_map[m]
        if clf_key not in best_params_all or reg_key not in best_params_all:
            logger.warning("No params found for %s — skipping full eval", m)
            continue

        results = _full_eval_tuned(
            model_labels[m],
            clf_cls, best_params_all[clf_key],
            reg_cls, best_params_all[reg_key],
            X, y_cls, y_reg, groups, le, numeric_features, t_values,
        )
        final_rows.append((
            "tuned_fe", model_labels[m],
            results["accuracy"], results["macro_f1"],
            results["Low_f1"], results["Medium_f1"], results["High_f1"],
            results["blast_mae"], results["blast_rmse"],
        ))
        all_t_rows.extend(results["t_rows"])
        logger.info(
            "  %s tuned → accuracy=%.3f  macro-F1=%.3f  "
            "[Low=%.3f  Med=%.3f  High=%.3f]  MAE=%.3f",
            model_labels[m],
            results["accuracy"], results["macro_f1"],
            results["Low_f1"], results["Medium_f1"], results["High_f1"],
            results["blast_mae"],
        )

    final_df = pd.DataFrame(final_rows, columns=PRIOR_COLS)
    final_df.to_csv(RESULTS_DIR / "final_results.csv", index=False)
    logger.info("\nFull three-stage comparison:\n%s", final_df.to_string(index=False))

    if all_t_rows:
        t_df = pd.DataFrame(all_t_rows)
        t_df.to_csv(RESULTS_DIR / "accuracy_vs_time_tuned.csv", index=False)
        logger.info("\nTime-into-fault (tuned models):\n%s", t_df.to_string(index=False))

    logger.info("\nDone. Models saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--data",      type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model",     choices=["rf", "lgbm", "all"], default="all")
    parser.add_argument("--trials",    type=int, default=N_TRIALS)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Optuna; load best_params.json and run full evaluation only")
    args = parser.parse_args()

    models = ["rf", "lgbm"] if args.model == "all" else [args.model]
    main(args.data, models, args.trials, eval_only=args.eval_only)
