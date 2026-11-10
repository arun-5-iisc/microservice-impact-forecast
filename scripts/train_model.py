"""
Untuned models (random forest, XGBoost and LightGBM with default settings) for both tasks, severity
classification and blast-radius regression from the early snapshot, compared with no-ML baselines on the
same folds. All three classifiers use balanced class weights. Baselines:
  RuleBaseline        only the latency thresholds, on the early snapshot (the midterm's baseline)
  FullRuleBaseline    the whole labelling rule on the early snapshot ("severity stays as it is now")
  EarlyBlastBaseline  final blast radius = blast radius at T
  NoSpreadBaseline    final blast radius = 0 
Cross-validation is GroupKFold grouped by experiment setting; the holdout runs are left out. Engineered
features are used when the dataset has them. Outputs go to models/<stage>/ and results/<stage>/.

Usage:
  python scripts/train_model.py --data dataset/processed/training_dataset.csv --stage baseline
  python scripts/train_model.py --data dataset/processed/training_dataset_fe.csv --stage fe
"""
import argparse
import logging
import sys
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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import _compute_severity   # the labelling rule - single source of truth

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
    "latency_observable",
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
    "service_upstream_count",
    "service_is_gateway",
    "fault_severity_ratio",
    "system_error_rate",
    "system_latency_ratio",
]

SEVERITY_ORDER = ["Low", "Medium", "High"]

# Thresholds from preprocess.py - used by the rule-based baseline
LATENCY_DELTA_HIGH   = 1.0
LATENCY_DELTA_MEDIUM = 0.20


# Preprocessing pipeline (built dynamically to match available columns)

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


# Model definitions

def _classifiers():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42,
            verbosity=0,
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


# Feature importance (works across all three model families)

def _get_importances(pipe, numeric_features: list[str]):
    estimator = pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    feat_names = numeric_features + CATEGORICAL_FEATURES
    imp = estimator.feature_importances_
    if len(imp) != len(feat_names):
        feat_names = [f"f{i}" for i in range(len(imp))]
    return pd.Series(imp, index=feat_names).sort_values(ascending=False)


# Rule-based baseline (no ML)

def latency_rule_severity(df: pd.DataFrame) -> np.ndarray:
    """
    Latency-only rule: the label's latency thresholds applied to the EARLY
    target_latency_delta. NaN latency (uninstrumented services) -> Low.
    Kept for comparison with the midterm; it ignores the replica, restart and
    error-rate rules, so it is a weak reference.
    """
    def _rule(v):
        if pd.isna(v) or v <= LATENCY_DELTA_MEDIUM:
            return "Low"
        if v <= LATENCY_DELTA_HIGH:
            return "Medium"
        return "High"

    return df["target_latency_delta"].map(_rule).values


def full_rule_severity(df: pd.DataFrame) -> np.ndarray:
    """
    Full-rule (persistence) baseline: the complete labelling rule from
    preprocess.py applied to the EARLY snapshot - "the final severity will be
    what it is now". This is effectively reactive alerting, the fair no-ML
    reference: a forecasting model adds value only where it beats this.
    """
    nz = lambda v: None if pd.isna(v) else v
    cols = ["target_latency_delta", "target_error_rate", "target_restart_count", "target_replicas"]
    return np.array([_compute_severity(*(nz(v) for v in row))
                     for row in df[cols].itertuples(index=False)])


def early_blast_radius(df: pd.DataFrame) -> np.ndarray:
    """"final = early" baseline: predict the final blast radius as the blast radius at T."""
    return df["blast_radius_early"].fillna(0).values.astype(float)


def no_spread_baseline(df: pd.DataFrame) -> np.ndarray:
    """"always 0" baseline: predict that the change spreads to no other service."""
    return np.zeros(len(df), dtype=float)


def baseline_cv_report(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    """
    Score a rule's predictions on the same GroupKFold folds, with the same
    aggregation as _evaluate_classifier (per-fold F1 averaged, accuracy pooled),
    so baseline and model numbers are directly comparable. Rules have no
    parameters, so the folds only define how the scores are averaged.
    """
    fold_reports = [
        classification_report(y_true[te], y_pred[te], labels=SEVERITY_ORDER,
                              target_names=SEVERITY_ORDER, output_dict=True, zero_division=0)
        for _, te in GroupKFold(n_splits=N_FOLDS).split(y_true, y_true, groups)
    ]
    agg = {cls: {m: np.mean([r[cls][m] for r in fold_reports])
                 for m in ("precision", "recall", "f1-score")}
           for cls in SEVERITY_ORDER + ["macro avg"]}
    agg["accuracy"] = accuracy_score(y_true, y_pred)
    return agg


def baseline_cv_regression(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    """Regression counterpart of baseline_cv_report, matching _evaluate_regressor's metrics."""
    mae, rmse = [], []
    for _, te in GroupKFold(n_splits=N_FOLDS).split(y_true, y_true, groups):
        mae.append(mean_absolute_error(y_true[te], y_pred[te]))
        rmse.append(np.sqrt(mean_squared_error(y_true[te], y_pred[te])))
    return {"mae_mean": np.mean(mae), "mae_std": np.std(mae),
            "rmse_mean": np.mean(rmse), "rmse_std": np.std(rmse)}


# Cross-validated evaluation (no retrain - fit on each fold)

def setting_groups(df: pd.DataFrame) -> np.ndarray:
    """CV group per experiment setting, so repeat runs of a setting never split across folds."""
    return (df["service"] + "|" + df["fault_type"] + "|" + df["fault_value"].astype(str)
            + "|" + df["traffic_level"]).values


def _evaluate_classifier(name, estimator, X, y_encoded, groups, le, numeric_features,
                         balanced_weights: bool = False):
    """
    balanced_weights: fit with "balanced" sample weights computed on each training fold
    (and on the full data for the refit). Used for XGBoost, which has no class_weight
    option, so it is weighted like the other classifiers here and like the tuned model.
    """
    def _fit_kwargs(y):
        return {"model__sample_weight": compute_sample_weight("balanced", y)} if balanced_weights else {}

    pipe = Pipeline([
        ("pre", _build_preprocessor(numeric_features)),
        ("model", estimator),
    ])
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_reports = []
    # Store OOF predictions by original row index to avoid ordering bugs
    oof_preds = {}
    oof_true  = {}

    for _, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_encoded[train_idx], y_encoded[test_idx]

        pipe.fit(X_tr, y_tr, **_fit_kwargs(y_tr))
        y_pred = pipe.predict(X_te)

        for i, idx in enumerate(test_idx):
            oof_preds[idx] = y_pred[i]
            oof_true[idx]  = y_te[i]

        report = classification_report(
            y_te, y_pred,
            labels=np.arange(len(le.classes_)),   # a fold may lack a class
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
    pipe.fit(X, y_encoded, **_fit_kwargs(y_encoded))
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

    for _, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
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


# Main

def main(data_path: Path, stage: str):
    models_dir  = MODELS_DIR  / stage
    results_dir = RESULTS_DIR / stage
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stage: %s  →  models=%s  results=%s", stage, models_dir, results_dir)
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset: %d rows, %d experiments", len(df), df["experiment_id"].nunique())

    # the holdout is left out here; it is scored only after tuning (tune_model.py)
    holdout_path = REPO_ROOT / "dataset" / "processed" / "holdout_ids.txt"
    if holdout_path.exists():
        holdout_ids = set(holdout_path.read_text().strip().splitlines())
        df_train    = df[~df["experiment_id"].isin(holdout_ids)].reset_index(drop=True)
        logger.info(
            "Holdout locked: using %d training experiments (%d rows)  "
            "(%d holdout experiments excluded)",
            df_train["experiment_id"].nunique(), len(df_train),
            df["experiment_id"].isin(holdout_ids).sum() // 4,
        )
    else:
        logger.warning(
            "holdout_ids.txt not found — using all data for training. "
            "Run: python scripts/create_holdout.py"
        )
        df_train = df.reset_index(drop=True)

    # Detect which feature columns are present (FE dataset has extra cols)
    numeric_features = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in df_train.columns]
    all_features     = numeric_features + CATEGORICAL_FEATURES
    is_fe_dataset    = any(c in df_train.columns for c in FE_NUMERIC_FEATURES)
    logger.info(
        "Feature set: %d numeric + %d categorical (%s)",
        len(numeric_features), len(CATEGORICAL_FEATURES),
        "FE dataset" if is_fe_dataset else "baseline dataset",
    )

    X_train  = df_train[all_features]
    groups   = setting_groups(df_train)

    # Task 1: Severity Classification
    logger.info("\n%s\nTask 1: Severity Classification\n%s", "="*55, "="*55)

    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_ORDER)
    y_cls = le.transform(df_train["severity"])

    clf_rows = []
    feat_imp_cls = {}
    best_oof_preds = None
    best_oof_true  = None

    # Rule-based baselines (no ML), scored on the same folds as the models
    for rule_name, rule_pred in (("RuleBaseline", latency_rule_severity(df_train)),
                                 ("FullRuleBaseline", full_rule_severity(df_train))):
        rule_report = baseline_cv_report(df_train["severity"].values, rule_pred, groups)
        rule_row = {"model": rule_name}
        for cls in SEVERITY_ORDER:
            rule_row[f"{cls}_f1"] = round(rule_report[cls]["f1-score"], 4)
        rule_row["macro_f1"] = round(rule_report["macro avg"]["f1-score"], 4)
        rule_row["accuracy"] = round(rule_report["accuracy"], 4)
        clf_rows.append(rule_row)
        logger.info(
            "  %s → accuracy=%.3f  macro-F1=%.3f  [Low=%.3f  Med=%.3f  High=%.3f]",
            rule_name, rule_report["accuracy"], rule_report["macro avg"]["f1-score"],
            rule_report["Low"]["f1-score"], rule_report["Medium"]["f1-score"],
            rule_report["High"]["f1-score"],
        )

    for name, estimator in _classifiers().items():
        logger.info("Training %s classifier...", name)
        # XGBoost has no class_weight option, so it gets the same balanced weights as sample weights
        pipe, report, imp, preds, true = _evaluate_classifier(
            name, estimator, X_train, y_cls, groups, le, numeric_features,
            balanced_weights=(name == "XGBoost"),
        )
        joblib.dump(pipe, models_dir / f"clf_{name.lower()}.joblib")
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
    clf_df.to_csv(results_dir / "classification_report.csv", index=False)
    logger.info("\nClassification results (5-fold CV on training set):\n%s",
                clf_df.to_string(index=False))

    # Task 2: Blast Radius Regression
    logger.info("\n%s\nTask 2: Blast Radius Regression\n%s", "="*55, "="*55)

    y_reg = df_train["blast_radius_final"].values.astype(float)

    reg_rows = []
    feat_imp_reg = {}

    for name, estimator in _regressors().items():
        logger.info("Training %s regressor...", name)
        pipe, metrics, imp = _evaluate_regressor(
            name, estimator, X_train, y_reg, groups, numeric_features
        )
        joblib.dump(pipe, models_dir / f"reg_{name.lower()}.joblib")

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

    base_metrics = baseline_cv_regression(y_reg, early_blast_radius(df_train), groups)
    reg_rows.append({"model": "EarlyBlastBaseline", **{k: round(v, 4) for k, v in base_metrics.items()}})
    logger.info("  EarlyBlastBaseline (final = early) → MAE=%.3f±%.3f  RMSE=%.3f±%.3f",
                base_metrics["mae_mean"], base_metrics["mae_std"],
                base_metrics["rmse_mean"], base_metrics["rmse_std"])
    zero_metrics = baseline_cv_regression(y_reg, no_spread_baseline(df_train), groups)
    reg_rows.append({"model": "NoSpreadBaseline", **{k: round(v, 4) for k, v in zero_metrics.items()}})
    logger.info("  NoSpreadBaseline (always 0)       → MAE=%.3f±%.3f  RMSE=%.3f±%.3f",
                zero_metrics["mae_mean"], zero_metrics["mae_std"],
                zero_metrics["rmse_mean"], zero_metrics["rmse_std"])

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(results_dir / "regression_report.csv", index=False)
    logger.info("\nRegression results:\n%s", reg_df.to_string(index=False))

    # Feature Importance
    logger.info("\n%s\nFeature Importance (Classification)\n%s", "="*55, "="*55)

    imp_records = []
    for model_name, imp in feat_imp_cls.items():
        for feat, score in imp.items():
            imp_records.append({"model": model_name, "feature": feat, "importance": round(score, 6)})

    imp_df = pd.DataFrame(imp_records)
    imp_df.to_csv(results_dir / "feature_importance.csv", index=False)

    if "RandomForest" in feat_imp_cls:
        top10 = feat_imp_cls["RandomForest"].head(10)
        logger.info("\nTop 10 features (RandomForest classifier):\n%s", top10.to_string())

    # Accuracy vs Time into Fault (OOF on training set)
    logger.info("\n%s\nAccuracy vs Time-into-fault (LightGBM OOF)\n%s", "="*55, "="*55)

    t_rows = []
    t_values = df_train["time_into_fault_s"].values
    for T in [15, 30, 45, 60]:
        mask = t_values == T
        if mask.sum() == 0:
            continue
        acc = accuracy_score(best_oof_true[mask], best_oof_preds[mask])
        f1  = f1_score(best_oof_true[mask], best_oof_preds[mask],
                       average="macro", zero_division=0)
        t_rows.append({"time_into_fault_s": T, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        logger.info("  T=%2ds → accuracy=%.3f  macro-F1=%.3f", T, acc, f1)

    pd.DataFrame(t_rows).to_csv(results_dir / "accuracy_vs_time.csv", index=False)

    logger.info("\nAll outputs saved to %s/ and %s/", results_dir, models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train impact forecasting models")
    parser.add_argument("--data",  type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage", type=str,  default=None,
                        help="Output subdirectory name (e.g. baseline, fe). "
                             "Auto-detected from filename if omitted.")
    args = parser.parse_args()
    stage = args.stage or ("fe" if "fe" in args.data.stem else "baseline")
    main(args.data, stage)
