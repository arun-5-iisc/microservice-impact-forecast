"""
Optuna tuning of the severity classifier or the blast-radius regressor (random forest, XGBoost or LightGBM).

Cross-validation is GroupKFold grouped by experiment setting, so a run's 4 snapshots and the repeats of a
setting never land on both sides of a split. Objective: macro-F1 for severity, MAE for blast radius. The best
model is then refitted, compared with the no-ML baselines and the untuned models, and scored once on the
holdout. Outputs go to models/<stage>/ and results/<stage>/.

Usage (stage 7 of run_pipeline.sh):
  python scripts/tune_model.py --data dataset/processed/training_dataset.csv --task clf --clf-model xgb --stage baseline_clf_xgb
  python scripts/tune_model.py --data dataset/processed/training_dataset_fe.csv --task reg --reg-model rf --stage fe_reg_rf
  (add --eval-only to evaluate again with the saved best_params.json)
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
    mean_squared_error,
    r2_score,
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
    baseline_cv_regression,
    baseline_cv_report,
    early_blast_radius,
    full_rule_severity,
    no_spread_baseline,
    setting_groups,
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


# Optuna objectives

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


# Generic tune + refit helper

def _tune(name: str, objective_fn, direction: str, n_trials: int,
          X, y, groups, numeric_features):
    logger.info("Tuning %s (%s, %d trials)…", name, direction, n_trials)
    t0 = time.time()

    def _wrapped(trial):
        return objective_fn(trial, X, y, groups, numeric_features)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
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

    Uses optuna.visualization.matplotlib - no kaleido or browser required.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )

        plots_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace("-", "_")

        ax = plot_optimization_history(study)
        ax.set_title(f"{name} — Optuna Convergence")
        ax.figure.savefig(plots_dir / f"convergence_{safe_name}.png",
                          dpi=150, bbox_inches="tight")
        plt.close("all")

        ax = plot_param_importances(study)
        ax.set_title(f"{name} — Hyperparameter Importance")
        ax.figure.savefig(plots_dir / f"param_importance_{safe_name}.png",
                          dpi=150, bbox_inches="tight")
        plt.close("all")

        logger.info("  Plots saved → %s/{convergence,param_importance}_%s.png",
                    plots_dir, safe_name)
    except Exception as exc:
        import traceback
        logger.warning("  Plot export failed (%s): %s\n%s",
                       name, exc, traceback.format_exc())


def _save_trial_history(study: optuna.Study, direction: str, path: Path) -> None:
    """Save per-trial values and running best to CSV for later convergence plotting."""
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df = df[df["state"] == "COMPLETE"][["number", "value"]].copy()
    if direction == "maximize":
        df["best_so_far"] = df["value"].cummax()
    else:
        df["best_so_far"] = df["value"].cummin()
    df.to_csv(path, index=False)
    logger.info("  Trial history → %s", path)


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


# Full OOF evaluation for tuned models

PRIOR_COLS = ["stage", "model", "accuracy", "macro_f1", "Low_f1",
              "Medium_f1", "High_f1", "blast_mae", "blast_rmse"]

def _clf_comparison_row(split: str, model: str, report: dict, accuracy: float) -> dict:
    row = {"split": split, "task": "severity", "model": model,
           "accuracy": round(accuracy, 4), "macro_f1": round(report["macro avg"]["f1-score"], 4)}
    for cls in SEVERITY_ORDER:
        row[f"{cls}_f1"] = round(report[cls]["f1-score"], 4)
    return row


def _reg_comparison_row(split: str, model: str, mae: float, rmse: float, r2: float = np.nan) -> dict:
    return {"split": split, "task": "blast_radius", "model": model,
            "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def _forecasting_gain(split: str, true_lbl, model_lbl, base_lbl, exp_ids) -> list:
    """
    Accuracy split by whether the severity changes after the snapshot. Where it
    changes, the full-rule baseline is wrong by definition, so a model's accuracy
    on those rows is where forecasting beats reactive alerting.
    """
    changed = base_lbl != true_lbl
    rows = []
    for subset, mask in (("all", np.ones(len(true_lbl), dtype=bool)),
                         ("changed_after_snapshot", changed), ("unchanged", ~changed)):
        if mask.sum() == 0:
            continue
        rows.append({
            "split": split, "subset": subset, "rows": int(mask.sum()),
            "experiments": len(set(exp_ids[mask])),
            "model_accuracy":    round(float((model_lbl[mask] == true_lbl[mask]).mean()), 4),
            "baseline_accuracy": round(float((base_lbl[mask] == true_lbl[mask]).mean()), 4),
        })
    return rows


def _load_prior_results() -> list[tuple]:
    """Results of the untuned models (train_model.py stages 'baseline' and 'fe'), to show the gain
    from tuning. Raises RuntimeError if they haven't been run yet."""
    rows = []
    missing = []
    for stage_key in ("baseline", "fe"):
        clf_path = RESULTS_DIR / stage_key / "classification_report.csv"
        reg_path = RESULTS_DIR / stage_key / "regression_report.csv"
        if not clf_path.exists() or not reg_path.exists():
            missing.append(stage_key)
            continue
        clf_df = pd.read_csv(clf_path)
        reg_df = pd.read_csv(reg_path)

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

    if missing:
        raise RuntimeError(
            f"Stage {missing} result CSVs not found. "
            "Run train_model.py --stage baseline and --stage fe before tuning."
        )

    logger.info("Loaded %d result rows of the untuned models", len(rows))
    return rows


# Per-model dispatch tables and param finalisation

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


def _oof_regression_by_time(estimator, X, y, groups, t_values, numeric_features):
    """
    Out-of-fold MAE/RMSE of the tuned regressor per snapshot time. train_model._evaluate_regressor()
    only returns fold means, not row-level predictions, so this reruns the same 5-fold split.
    """
    pipe = Pipeline([("pre", _build_preprocessor(numeric_features)), ("model", estimator)])
    gkf  = GroupKFold(n_splits=N_FOLDS)

    oof_preds = np.empty(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        oof_preds[test_idx] = pipe.predict(X.iloc[test_idx])

    rows = []
    for T in [15, 30, 45, 60]:
        mask = t_values == T
        if mask.sum() == 0:
            continue
        mae  = mean_absolute_error(y[mask], oof_preds[mask])
        rmse = np.sqrt(mean_squared_error(y[mask], oof_preds[mask]))
        rows.append({"time_into_fault_s": T, "mae": round(mae, 4), "rmse": round(rmse, 4)})
    return rows


# Main: tunes the classifier and/or the regressor, each on its own

def main(data_path: Path, stage: str, clf_model: str, reg_model: str,
         n_trials: int, eval_only: bool = False, task: str = "both"):
    models_dir  = MODELS_DIR  / stage
    results_dir = RESULTS_DIR / stage
    plots_dir   = results_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s  [stage=%s]", data_path, stage)
    logger.info("Tuning: %s", ", ".join(
        ([f"classifier {_LABELS[clf_model]}"] if task in ("both", "clf") else [])
        + ([f"regressor {_LABELS[reg_model]}"] if task in ("both", "reg") else [])))
    df = pd.read_csv(data_path)
    logger.info("Dataset: %d rows, %d experiments", len(df), df["experiment_id"].nunique())

    # Train / holdout split
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
    groups = setting_groups(df_train)

    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_ORDER)
    y_cls = le.transform(df_train["severity"])
    y_reg = df_train["blast_radius_final"].values.astype(float)

    best_params_all = {}
    report_rows     = []

    # untuned results of the same feature set, to report the gain from tuning
    prior_stage  = "baseline" if "baseline" in stage else "fe"
    all_prior    = _load_prior_results()
    prior_lookup = {
        row[1]: {"macro_f1": row[3], "mae": row[7]}
        for row in all_prior
        if row[0] == prior_stage
    }

    # Load existing params if eval-only
    params_path = results_dir / "best_params.json"
    if eval_only:
        if not params_path.exists():
            logger.error("--eval-only requires %s to exist. Run tuning first.", params_path)
            return
        with open(params_path) as f:
            best_params_all = json.load(f)
        logger.info("Loaded best params from %s", params_path)

    # Classification pipeline: tune clf_model
    clf_key = f"{clf_model}_clf"
    if task in ("both", "clf"):
        logger.info("\n%s\nClassification Pipeline — tuning %s\n%s",
                    "="*55, _LABELS[clf_model], "="*55)
    if task in ("both", "clf") and not eval_only:
        val, params, study = _tune(
            f"{clf_model.upper()}-clf", _CLF_OBJ[clf_model], "maximize", n_trials,
            X, y_cls, groups, numeric_features,
        )
        _save_study_plots(study, f"{clf_model}-clf", plots_dir)
        _save_trial_history(study, "maximize", results_dir / f"trial_history_{clf_model}_clf.csv")
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

    # Regression pipeline: tune reg_model
    reg_key = f"{reg_model}_reg"
    if task in ("both", "reg"):
        logger.info("\n%s\nRegression Pipeline — tuning %s\n%s",
                    "="*55, _LABELS[reg_model], "="*55)
    if task in ("both", "reg") and not eval_only:
        val, params, study = _tune(
            f"{reg_model.upper()}-reg", _REG_OBJ[reg_model], "minimize", n_trials,
            X, y_reg, groups, numeric_features,
        )
        _save_study_plots(study, f"{reg_model}-reg", plots_dir)
        _save_trial_history(study, "minimize", results_dir / f"trial_history_{reg_model}_reg.csv")
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

    # Save tuning report
    if not eval_only:
        with open(params_path, "w") as f:
            json.dump(best_params_all, f, indent=2)
        pd.DataFrame(report_rows).to_csv(results_dir / "tuning_report.csv", index=False)
        logger.info("Best params → %s", params_path)

    # Full OOF evaluation
    logger.info("\n%s\nFull OOF Evaluation\n%s", "="*55, "="*55)

    final_rows  = list(all_prior)
    all_t_rows  = []
    oof_records = []
    comparison_rows = []   # model vs no-ML baselines -> baseline_comparison.csv
    gain_rows       = []   # accuracy where severity changes after the snapshot -> forecasting_gain.csv
    t_values    = df_train["time_into_fault_s"].values

    # Classification: evaluate tuned clf_model
    if clf_key in best_params_all:
        logger.info("  OOF eval — %s classifier (tuned)…", _LABELS[clf_model])
        _, clf_report, _, oof_preds, oof_true = _evaluate_classifier(
            _LABELS[clf_model],
            _CLF_CLS[clf_model](**best_params_all[clf_key]),
            X, y_cls, groups, le, numeric_features,
            # same class weighting as the tuning objective and the saved model (XGBoost)
            balanced_weights=_get_clf_fit_params(clf_model, y_cls) is not None,
        )
        true_lbl  = le.classes_[oof_true]
        model_lbl = le.classes_[oof_preds]
        base_lbl  = full_rule_severity(df_train)
        assert (true_lbl == df_train["severity"].values).all(), "OOF rows out of order"
        base_report = baseline_cv_report(true_lbl, base_lbl, groups)
        for T in [15, 30, 45, 60]:
            mask = t_values == T
            if mask.sum() == 0:
                continue
            acc = accuracy_score(oof_true[mask], oof_preds[mask])
            f1  = f1_score(oof_true[mask], oof_preds[mask], average="macro", zero_division=0)
            b_acc = accuracy_score(true_lbl[mask], base_lbl[mask])
            b_f1  = f1_score(true_lbl[mask], base_lbl[mask], average="macro", zero_division=0)
            all_t_rows += [
                {"model": _LABELS[clf_model], "time_into_fault_s": T,
                 "accuracy": round(acc, 4), "macro_f1": round(f1, 4)},
                {"model": "FullRuleBaseline", "time_into_fault_s": T,
                 "accuracy": round(b_acc, 4), "macro_f1": round(b_f1, 4)},
            ]
            logger.info("    T=%2ds → accuracy=%.3f  macro-F1=%.3f   (full-rule baseline %.3f / %.3f)",
                        T, acc, f1, b_acc, b_f1)
        for i, (pred, true) in enumerate(zip(oof_preds, oof_true)):
            oof_records.append({
                "model":             _LABELS[clf_model],
                "true_label":        le.classes_[true],
                "pred_label":        le.classes_[pred],
                "baseline_pred":     base_lbl[i],
                "experiment_id":     df_train["experiment_id"].iat[i],
                "time_into_fault_s": int(t_values[i]),
            })
        comparison_rows += [
            _clf_comparison_row("cv", _LABELS[clf_model], clf_report, clf_report["accuracy"]),
            _clf_comparison_row("cv", "FullRuleBaseline", base_report, base_report["accuracy"]),
        ]
        gain_rows += _forecasting_gain("cv", true_lbl, model_lbl, base_lbl,
                                       df_train["experiment_id"].values)
        logger.info("  FullRuleBaseline (cv) → accuracy=%.3f  macro-F1=%.3f",
                    base_report["accuracy"], base_report["macro avg"]["f1-score"])
        # classifier row: the blast-radius columns stay NaN
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
    mae_t_rows = []
    if reg_key in best_params_all:
        logger.info("  OOF eval — %s regressor (tuned)…", _LABELS[reg_model])
        _, reg_metrics, _ = _evaluate_regressor(
            _LABELS[reg_model],
            _REG_CLS[reg_model](**best_params_all[reg_key]),
            X, y_reg, groups, numeric_features,
        )
        # regressor row: the severity columns stay NaN
        final_rows.append((
            stage, _LABELS[reg_model],
            np.nan, np.nan, np.nan, np.nan, np.nan,
            round(reg_metrics["mae_mean"], 4),
            round(reg_metrics["rmse_mean"], 4),
        ))
        logger.info("  %s reg tuned → MAE=%.4f±%.4f  RMSE=%.4f±%.4f",
                    _LABELS[reg_model],
                    reg_metrics["mae_mean"],  reg_metrics["mae_std"],
                    reg_metrics["rmse_mean"], reg_metrics["rmse_std"])
        early = early_blast_radius(df_train)
        zero = no_spread_baseline(df_train)
        base_reg = baseline_cv_regression(y_reg, early, groups)
        zero_reg = baseline_cv_regression(y_reg, zero, groups)
        comparison_rows += [
            _reg_comparison_row("cv", _LABELS[reg_model], reg_metrics["mae_mean"], reg_metrics["rmse_mean"]),
            _reg_comparison_row("cv", "EarlyBlastBaseline", base_reg["mae_mean"], base_reg["rmse_mean"]),
            _reg_comparison_row("cv", "NoSpreadBaseline", zero_reg["mae_mean"], zero_reg["rmse_mean"]),
        ]
        logger.info("  EarlyBlastBaseline (cv) → MAE=%.4f  RMSE=%.4f", base_reg["mae_mean"], base_reg["rmse_mean"])
        logger.info("  NoSpreadBaseline   (cv) → MAE=%.4f  RMSE=%.4f", zero_reg["mae_mean"], zero_reg["rmse_mean"])
        for T in [15, 30, 45, 60]:
            mask = t_values == T
            if mask.sum():
                for name, pred in (("EarlyBlastBaseline", early), ("NoSpreadBaseline", zero)):
                    mae_t_rows.append({
                        "model": name, "time_into_fault_s": T,
                        "mae":  round(mean_absolute_error(y_reg[mask], pred[mask]), 4),
                        "rmse": round(float(np.sqrt(np.mean((y_reg[mask] - pred[mask]) ** 2))), 4),
                    })

        logger.info("  OOF eval by time-into-fault — %s regressor (tuned)…", _LABELS[reg_model])
        for row in _oof_regression_by_time(
            _REG_CLS[reg_model](**best_params_all[reg_key]),
            X, y_reg, groups, t_values, numeric_features,
        ):
            row["model"] = _LABELS[reg_model]
            mae_t_rows.append(row)
            logger.info("    T=%2ds → MAE=%.3f  RMSE=%.3f",
                        row["time_into_fault_s"], row["mae"], row["rmse"])

    # Save results
    final_df = pd.DataFrame(final_rows, columns=PRIOR_COLS)
    final_df.to_csv(results_dir / "final_results.csv", index=False)
    logger.info("\nFull three-stage comparison:\n%s", final_df.to_string(index=False))

    if mae_t_rows:
        pd.DataFrame(mae_t_rows).to_csv(results_dir / "mae_vs_time_tuned.csv", index=False)

    if all_t_rows:
        pd.DataFrame(all_t_rows).to_csv(results_dir / "accuracy_vs_time_tuned.csv", index=False)

    if oof_records:
        pd.DataFrame(oof_records).to_csv(results_dir / "oof_predictions.csv", index=False)
        logger.info("OOF predictions → %s", results_dir / "oof_predictions.csv")

    # Holdout: scored once, after tuning
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
            true_test = le.inverse_transform(y_cls_test)
            pred_test = le.inverse_transform(y_pred.astype(int))
            base_test = full_rule_severity(df_test)
            base_rep  = classification_report(true_test, base_test, labels=SEVERITY_ORDER,
                                              target_names=SEVERITY_ORDER, output_dict=True,
                                              zero_division=0)
            base_acc  = accuracy_score(true_test, base_test)
            h_rows = []
            for name, rep, a in ((_LABELS[clf_model], report, acc), ("FullRuleBaseline", base_rep, base_acc)):
                row = {"model": name, "accuracy": round(a, 4), "macro_f1": round(rep["macro avg"]["f1-score"], 4)}
                for cls in SEVERITY_ORDER:
                    row[f"{cls}_f1"] = round(rep[cls]["f1-score"], 4)
                h_rows.append(row)
            # the model stays on the first row - visualize.py reads row 0
            pd.DataFrame(h_rows).to_csv(results_dir / "holdout_report.csv", index=False)
            comparison_rows += [
                _clf_comparison_row("holdout", _LABELS[clf_model], report, acc),
                _clf_comparison_row("holdout", "FullRuleBaseline", base_rep, base_acc),
            ]
            gain_rows += _forecasting_gain("holdout", true_test, pred_test, base_test,
                                           df_test["experiment_id"].values)
            logger.info("  FullRuleBaseline holdout → accuracy=%.3f  macro-F1=%.3f",
                        base_acc, base_rep["macro avg"]["f1-score"])
            t_test = df_test["time_into_fault_s"].values
            pd.DataFrame({
                "model":             _LABELS[clf_model],
                "true_label":        true_test,
                "pred_label":        pred_test,
                "baseline_pred":     base_test,
                "experiment_id":     df_test["experiment_id"].values,
                "time_into_fault_s": t_test,
            }).to_csv(results_dir / "holdout_predictions.csv", index=False)
            logger.info(
                "  %s clf holdout → accuracy=%.3f  macro-F1=%.3f  "
                "[Low=%.3f  Med=%.3f  High=%.3f]",
                _LABELS[clf_model], acc,
                report["macro avg"]["f1-score"],
                report["Low"]["f1-score"], report["Medium"]["f1-score"],
                report["High"]["f1-score"],
            )

            t_rows = []
            for T in [15, 30, 45, 60]:
                mask = t_test == T
                if mask.sum() == 0:
                    continue
                t_acc = accuracy_score(y_cls_test[mask], y_pred[mask])
                t_f1  = f1_score(y_cls_test[mask], y_pred[mask], average="macro", zero_division=0)
                b_acc = accuracy_score(true_test[mask], base_test[mask])
                b_f1  = f1_score(true_test[mask], base_test[mask], average="macro", zero_division=0)
                t_rows += [
                    {"model": _LABELS[clf_model], "time_into_fault_s": T,
                     "accuracy": round(t_acc, 4), "macro_f1": round(t_f1, 4)},
                    {"model": "FullRuleBaseline", "time_into_fault_s": T,
                     "accuracy": round(b_acc, 4), "macro_f1": round(b_f1, 4)},
                ]
                logger.info("    holdout T=%2ds → accuracy=%.3f  macro-F1=%.3f   (full-rule baseline %.3f / %.3f)",
                            T, t_acc, t_f1, b_acc, b_f1)
            if t_rows:
                pd.DataFrame(t_rows).to_csv(results_dir / "accuracy_vs_time_holdout.csv", index=False)

        # Best regressor on holdout
        reg_path = models_dir / f"reg_{reg_model}_tuned.joblib"
        if reg_path.exists():
            pipe   = joblib.load(reg_path)
            y_pred = pipe.predict(X_test)
            mae  = mean_absolute_error(y_reg_test, y_pred)
            rmse = float(np.sqrt(np.mean((y_reg_test - y_pred) ** 2)))
            r2   = r2_score(y_reg_test, y_pred)
            early_test = early_blast_radius(df_test)
            b_mae  = mean_absolute_error(y_reg_test, early_test)
            b_rmse = float(np.sqrt(np.mean((y_reg_test - early_test) ** 2)))
            b_r2   = r2_score(y_reg_test, early_test)
            zero_test = no_spread_baseline(df_test)
            z_mae  = mean_absolute_error(y_reg_test, zero_test)
            z_rmse = float(np.sqrt(np.mean((y_reg_test - zero_test) ** 2)))
            z_r2   = r2_score(y_reg_test, zero_test)
            # the model stays on the first row - visualize.py reads row 0
            pd.DataFrame([
                {"model": _LABELS[reg_model], "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
                {"model": "EarlyBlastBaseline", "mae": round(b_mae, 4), "rmse": round(b_rmse, 4), "r2": round(b_r2, 4)},
                {"model": "NoSpreadBaseline", "mae": round(z_mae, 4), "rmse": round(z_rmse, 4), "r2": round(z_r2, 4)},
            ]).to_csv(results_dir / "holdout_regression_report.csv", index=False)
            comparison_rows += [
                _reg_comparison_row("holdout", _LABELS[reg_model], mae, rmse, r2),
                _reg_comparison_row("holdout", "EarlyBlastBaseline", b_mae, b_rmse, b_r2),
                _reg_comparison_row("holdout", "NoSpreadBaseline", z_mae, z_rmse, z_r2),
            ]
            t_test = df_test["time_into_fault_s"].values
            pd.DataFrame({
                "model":             _LABELS[reg_model],
                "true_blast_radius": y_reg_test,
                "pred_blast_radius": y_pred,
                "baseline_pred":     early_test,
                "severity":          df_test["severity"].values,
                "experiment_id":     df_test["experiment_id"].values,
                "time_into_fault_s": t_test,
            }).to_csv(results_dir / "holdout_regression_predictions.csv", index=False)
            logger.info("  %s reg holdout → MAE=%.4f  RMSE=%.4f  R²=%.4f   (final = early baseline MAE=%.4f, "
                        "always-0 baseline MAE=%.4f)", _LABELS[reg_model], mae, rmse, r2, b_mae, z_mae)

            t_rows = []
            for T in [15, 30, 45, 60]:
                mask = t_test == T
                if mask.sum() == 0:
                    continue
                t_mae  = mean_absolute_error(y_reg_test[mask], y_pred[mask])
                t_rmse = float(np.sqrt(np.mean((y_reg_test[mask] - y_pred[mask]) ** 2)))
                b_mae_t  = mean_absolute_error(y_reg_test[mask], early_test[mask])
                b_rmse_t = float(np.sqrt(np.mean((y_reg_test[mask] - early_test[mask]) ** 2)))
                z_mae_t  = mean_absolute_error(y_reg_test[mask], zero_test[mask])
                z_rmse_t = float(np.sqrt(np.mean((y_reg_test[mask] - zero_test[mask]) ** 2)))
                t_rows += [
                    {"model": _LABELS[reg_model], "time_into_fault_s": T,
                     "mae": round(t_mae, 4), "rmse": round(t_rmse, 4)},
                    {"model": "EarlyBlastBaseline", "time_into_fault_s": T,
                     "mae": round(b_mae_t, 4), "rmse": round(b_rmse_t, 4)},
                    {"model": "NoSpreadBaseline", "time_into_fault_s": T,
                     "mae": round(z_mae_t, 4), "rmse": round(z_rmse_t, 4)},
                ]
                logger.info("    holdout T=%2ds → MAE=%.3f  RMSE=%.3f   (final = early baseline MAE=%.3f)",
                            T, t_mae, t_rmse, b_mae_t)
            if t_rows:
                pd.DataFrame(t_rows).to_csv(results_dir / "mae_vs_time_holdout.csv", index=False)

    # Model vs no-ML baselines
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(results_dir / "baseline_comparison.csv", index=False)
        logger.info("\nModel vs no-ML baselines:\n%s", pd.DataFrame(comparison_rows).to_string(index=False))
    if gain_rows:
        pd.DataFrame(gain_rows).to_csv(results_dir / "forecasting_gain.csv", index=False)
        logger.info("\nAccuracy where severity changes after the snapshot:\n%s",
                    pd.DataFrame(gain_rows).to_string(index=False))

    logger.info("\nDone. Models → %s/", models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--data",      type=Path, default=DEFAULT_DATA,
                        help="Training dataset CSV")
    parser.add_argument("--stage",     type=str,  default=None,
                        help="Output sub-directory (auto-detected from --data if omitted)")
    parser.add_argument("--clf-model", dest="clf_model", choices=["rf", "xgb", "lgbm"],
                        default="xgb",
                        help="classifier to tune (default: xgb, the final model)")
    parser.add_argument("--reg-model", dest="reg_model", choices=["rf", "xgb", "lgbm"],
                        default="xgb",
                        help="regressor to tune (default: xgb, the final model)")
    parser.add_argument("--trials",    type=int, default=N_TRIALS)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Optuna; load best_params.json and run full evaluation only")
    parser.add_argument("--task", choices=["both", "clf", "reg"], default="both",
                        help="Which pipeline to tune: both (default), clf only, or reg only")
    args = parser.parse_args()

    stage = args.stage or (
        "baseline_tuned" if "fe" not in args.data.stem else "fe_tuned"
    )
    main(args.data, stage, args.clf_model, args.reg_model, args.trials, args.eval_only, args.task)
