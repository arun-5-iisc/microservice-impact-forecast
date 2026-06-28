"""
Train ML models for proactive deployment impact forecasting.

Tasks:
  1. Severity classification  — predict Low / Medium / High from early signals
  2. Blast radius regression  — predict final blast_radius_final from early signals

Models: RandomForest, XGBoost, LightGBM (all three for both tasks)

Evaluation uses GroupKFold grouped by experiment_id so all 4 T-snapshots
of the same experiment land in the same fold — prevents label leakage across
the time-into-fault dimension.

Works with both:
  - Baseline dataset (training_dataset.csv)         — only base numeric/categorical cols
  - FE dataset      (training_dataset_fe.csv)        — adds topology, log, ratio, system cols

Extra FE columns are used automatically when present; absent columns are silently skipped.

Outputs:
  models/    — saved .joblib files for each model
  results/   — classification_report.csv, regression_report.csv,
               feature_importance.csv, accuracy_vs_time.csv

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --data dataset/processed/training_dataset.csv
  python scripts/train_model.py --data dataset/processed/training_dataset_fe.csv
"""
import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"
MODELS_DIR   = REPO_ROOT / "models"
RESULTS_DIR  = REPO_ROOT / "results"

N_FOLDS = 5

CATEGORICAL_FEATURES = ["fault_type", "service", "traffic_level"]

# Base numeric features (always present in training_dataset.csv)
BASE_NUMERIC_FEATURES = [
    "fault_value_numeric",
    "time_into_fault_s",
    "target_cpu",
    "target_cpu_delta",
    "target_memory_bytes",
    "target_memory_delta",
    "target_replicas",
    "target_restart_count",
    "system_cpu_mean",
    "system_memory_mean",
    "target_request_rate",
    "target_error_rate",
    "target_latency_mean_ms",
    "target_latency_delta",
    "blast_radius_early",
    "max_trace_depth",
]

# Optional FE columns added by feature_engineer.py
FE_NUMERIC_FEATURES = [
    "service_depth",
    "service_downstream_count",
    "service_is_gateway",
    "fault_severity_ratio",
    "log_target_latency_mean_ms",
    "log_target_memory_bytes",
    "log_fault_value_numeric",
    "system_error_rate",
    "system_latency_ratio",
]

SEVERITY_ORDER = ["Low", "Medium", "High"]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline (built dynamically to match available columns)
# ─────────────────────────────────────────────────────────────────────────────

def _build_preprocessor(numeric_features: list[str]):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe,     numeric_features),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def _classifiers():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42,
            verbosity=0, use_label_encoder=False,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            class_weight="balanced", random_state=42,
            verbose=-1,
        ),
    }


def _regressors():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            random_state=42, verbose=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance (works across all three model families)
# ─────────────────────────────────────────────────────────────────────────────

def _get_importances(pipe, numeric_features: list[str]):
    estimator = pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    feat_names = numeric_features + CATEGORICAL_FEATURES
    imp = estimator.feature_importances_
    if len(imp) != len(feat_names):
        feat_names = [f"f{i}" for i in range(len(imp))]
    return pd.Series(imp, index=feat_names).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation (no retrain — fit on each fold)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_classifier(name, estimator, X, y_encoded, groups, le, numeric_features):
    pipe = Pipeline([
        ("pre", _build_preprocessor(numeric_features)),
        ("model", estimator),
    ])
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_reports = []
    # Store OOF predictions by original row index to avoid ordering bugs
    oof_preds = {}
    oof_true  = {}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_encoded[train_idx], y_encoded[test_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        for i, idx in enumerate(test_idx):
            oof_preds[idx] = y_pred[i]
            oof_true[idx]  = y_te[i]

        report = classification_report(
            y_te, y_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        fold_reports.append(report)

    # Reconstruct full OOF arrays in original row order
    sorted_idx = sorted(oof_preds)
    preds_arr  = np.array([oof_preds[i] for i in sorted_idx])
    true_arr   = np.array([oof_true[i]  for i in sorted_idx])

    # Aggregate metrics across folds
    agg = {}
    for cls in le.classes_:
        agg[cls] = {
            m: np.mean([r[cls][m] for r in fold_reports])
            for m in ("precision", "recall", "f1-score")
        }
    agg["macro avg"] = {
        m: np.mean([r["macro avg"][m] for r in fold_reports])
        for m in ("precision", "recall", "f1-score")
    }
    agg["accuracy"] = accuracy_score(true_arr, preds_arr)

    # Refit on full data for feature importance + saved model
    pipe.fit(X, y_encoded)
    imp = _get_importances(pipe, numeric_features)

    return pipe, agg, imp, preds_arr, true_arr


def _evaluate_regressor(name, estimator, X, y, groups, numeric_features):
    pipe = Pipeline([
        ("pre", _build_preprocessor(numeric_features)),
        ("model", estimator),
    ])
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_mae, fold_rmse = [], []
    all_preds, all_true = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        all_preds.extend(y_pred)
        all_true.extend(y_te)

        fold_mae.append(mean_absolute_error(y_te, y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_te, y_pred)))

    pipe.fit(X, y)
    imp = _get_importances(pipe, numeric_features)

    return pipe, {
        "mae_mean":  np.mean(fold_mae),
        "mae_std":   np.std(fold_mae),
        "rmse_mean": np.mean(fold_rmse),
        "rmse_std":  np.std(fold_rmse),
    }, imp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: Path):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset: %d rows, %d experiments", len(df), df["experiment_id"].nunique())

    # Detect which feature columns are present (FE dataset has extra cols)
    numeric_features = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in df.columns]
    all_features     = numeric_features + CATEGORICAL_FEATURES
    is_fe_dataset    = any(c in df.columns for c in FE_NUMERIC_FEATURES)
    logger.info(
        "Feature set: %d numeric + %d categorical (%s)",
        len(numeric_features), len(CATEGORICAL_FEATURES),
        "FE dataset" if is_fe_dataset else "baseline dataset",
    )

    X      = df[all_features]
    groups = df["experiment_id"].values

    # ── Task 1: Severity Classification ──────────────────────────────────────
    logger.info("\n%s\nTask 1: Severity Classification\n%s", "="*55, "="*55)

    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_ORDER)
    y_cls = le.transform(df["severity"])

    clf_rows = []
    feat_imp_cls = {}
    best_oof_preds = None   # OOF predictions from LightGBM (best macro-F1) for time analysis
    best_oof_true  = None

    for name, estimator in _classifiers().items():
        logger.info("Training %s classifier...", name)
        pipe, report, imp, preds, true = _evaluate_classifier(
            name, estimator, X, y_cls, groups, le, numeric_features
        )
        joblib.dump(pipe, MODELS_DIR / f"clf_{name.lower()}.joblib")
        # Keep LightGBM OOF preds for the time-into-fault analysis (best model)
        if name == "LightGBM":
            best_oof_preds = preds
            best_oof_true  = true

        row = {"model": name}
        for cls in SEVERITY_ORDER:
            row[f"{cls}_f1"] = round(report[cls]["f1-score"], 4)
        row["macro_f1"]  = round(report["macro avg"]["f1-score"], 4)
        row["accuracy"]  = round(report["accuracy"], 4)
        clf_rows.append(row)

        if imp is not None:
            feat_imp_cls[name] = imp

        logger.info(
            "  %s → accuracy=%.3f  macro-F1=%.3f  "
            "[Low=%.3f  Med=%.3f  High=%.3f]",
            name,
            report["accuracy"],
            report["macro avg"]["f1-score"],
            report["Low"]["f1-score"],
            report["Medium"]["f1-score"],
            report["High"]["f1-score"],
        )

    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(RESULTS_DIR / "classification_report.csv", index=False)
    logger.info("\nClassification results:\n%s", clf_df.to_string(index=False))

    # ── Task 2: Blast Radius Regression ──────────────────────────────────────
    logger.info("\n%s\nTask 2: Blast Radius Regression\n%s", "="*55, "="*55)

    y_reg = df["blast_radius_final"].values.astype(float)

    reg_rows = []
    feat_imp_reg = {}

    for name, estimator in _regressors().items():
        logger.info("Training %s regressor...", name)
        pipe, metrics, imp = _evaluate_regressor(name, estimator, X, y_reg, groups, numeric_features)
        joblib.dump(pipe, MODELS_DIR / f"reg_{name.lower()}.joblib")

        row = {"model": name, **{k: round(v, 4) for k, v in metrics.items()}}
        reg_rows.append(row)

        if imp is not None:
            feat_imp_reg[name] = imp

        logger.info(
            "  %s → MAE=%.3f±%.3f  RMSE=%.3f±%.3f",
            name,
            metrics["mae_mean"], metrics["mae_std"],
            metrics["rmse_mean"], metrics["rmse_std"],
        )

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(RESULTS_DIR / "regression_report.csv", index=False)
    logger.info("\nRegression results:\n%s", reg_df.to_string(index=False))

    # ── Feature Importance ────────────────────────────────────────────────────
    logger.info("\n%s\nFeature Importance (Classification)\n%s", "="*55, "="*55)

    imp_records = []
    for model_name, imp in feat_imp_cls.items():
        for feat, score in imp.items():
            imp_records.append({"model": model_name, "feature": feat, "importance": round(score, 6)})

    imp_df = pd.DataFrame(imp_records)
    imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # Print top 10 by RandomForest (most interpretable)
    if "RandomForest" in feat_imp_cls:
        top10 = feat_imp_cls["RandomForest"].head(10)
        logger.info("\nTop 10 features (RandomForest classifier):\n%s", top10.to_string())

    # ── Accuracy vs Time into Fault ───────────────────────────────────────────
    # Uses LightGBM OOF predictions (never seen training data) to avoid leakage.
    logger.info("\n%s\nAccuracy vs Time-into-fault (LightGBM OOF)\n%s", "="*55, "="*55)

    t_rows = []
    t_values = df["time_into_fault_s"].values
    for T in [15, 30, 45, 60]:
        mask = t_values == T
        if mask.sum() == 0:
            continue
        acc = accuracy_score(best_oof_true[mask], best_oof_preds[mask])
        f1  = f1_score(best_oof_true[mask], best_oof_preds[mask],
                       average="macro", zero_division=0)
        t_rows.append({"time_into_fault_s": T, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        logger.info("  T=%2ds → accuracy=%.3f  macro-F1=%.3f", T, acc, f1)

    pd.DataFrame(t_rows).to_csv(RESULTS_DIR / "accuracy_vs_time.csv", index=False)

    logger.info("\nAll outputs saved to %s/ and %s/", RESULTS_DIR, MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train impact forecasting models")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    main(args.data)
