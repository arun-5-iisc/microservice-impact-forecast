"""
Model explainability: SHAP values and permutation importance.

Loads a trained pipeline (.joblib), passes the training runs through its preprocessor (the
holdout in holdout_ids.txt is left out) and computes SHAP values, plus permutation importance
for the classifier.

Outputs, in results/<stage>/plots/:
  clf task:  shap_summary.png, shap_bar.png, shap_values.csv,
             permutation_importance.png, permutation_importance.csv
  reg task:  reg_shap_summary.png, reg_shap_bar.png, reg_shap_values.csv

Usage (stage 9 of run_pipeline.sh):
  python scripts/explain_model.py --model models/baseline_clf_xgb/clf_xgb_tuned.joblib \\
      --data dataset/processed/training_dataset.csv --stage baseline_clf_xgb --task clf
  python scripts/explain_model.py --model models/fe_reg_xgb/reg_xgb_tuned.joblib \\
      --data dataset/processed/training_dataset_fe.csv --stage fe_reg_xgb --task reg
"""
import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless - no display required
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_model import (
    BASE_NUMERIC_FEATURES,
    FE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SEVERITY_ORDER,
    setting_groups,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


# SHAP

def _shap_plots(pipe, X_transformed: np.ndarray, feature_names: list[str],
                plots_dir: Path, model_name: str) -> None:
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return

    estimator = pipe.named_steps["model"]
    logger.info("Computing SHAP values for %s…", model_name)

    try:
        explainer   = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as exc:
        logger.error("SHAP TreeExplainer failed: %s", exc)
        return

    # SHAP multiclass output varies by version:
    #   Old API: list of 2D arrays [(n_samples, n_features), ...] one per class
    #   New API: 3D numpy array (n_samples, n_features, n_classes)
    # Normalise to a list of 2D arrays so the rest of the code is uniform.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    if isinstance(shap_values, list):
        mean_abs    = [np.abs(sv).mean() for sv in shap_values]
        primary_idx = int(np.argmax(mean_abs))
        sv_primary  = shap_values[primary_idx]
        class_label = SEVERITY_ORDER[primary_idx]
        logger.info("  Multiclass SHAP — showing class '%s' (largest effect)", class_label)
    else:
        sv_primary  = shap_values
        primary_idx = 0
        class_label = "prediction"

    plots_dir.mkdir(parents=True, exist_ok=True)
    shap_df = pd.DataFrame(sv_primary, columns=feature_names)
    shap_df.to_csv(plots_dir / "shap_values.csv", index=False)

    # Beeswarm summary plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv_primary, X_transformed, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title(f"SHAP Summary — {model_name}  (class: {class_label})", fontsize=11)
    plt.tight_layout()
    path = plots_dir / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved → %s", path)

    # Bar plot (mean |SHAP|)
    # Multiclass: one combined chart, all classes shown together as stacked
    # per-class segments (not just the single "largest effect" class used
    # above for the beeswarm) - shap's own multiclass support for plot_type="bar".
    fig, ax = plt.subplots(figsize=(9, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                          plot_type="bar", class_names=SEVERITY_ORDER,
                          show=False, plot_size=None)
        plt.title(f"SHAP Feature Importance — {model_name}  (mean |SHAP|, all classes)",
                  fontsize=11)
    else:
        shap.summary_plot(sv_primary, X_transformed, feature_names=feature_names,
                          plot_type="bar", show=False, plot_size=None)
        plt.title(f"SHAP Feature Importance — {model_name}  (mean |SHAP|)", fontsize=11)
    plt.tight_layout()
    path = plots_dir / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved → %s", path)


# Regressor SHAP

def _reg_shap_plots(pipe, X_transformed: np.ndarray, feature_names: list[str],
                    plots_dir: Path, model_name: str) -> None:
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return

    estimator = pipe.named_steps["model"]
    logger.info("Computing SHAP values for regressor %s…", model_name)

    try:
        explainer   = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as exc:
        logger.error("SHAP TreeExplainer failed: %s", exc)
        return

    # Regression output is 2D (n_samples, n_features) - no class axis
    plots_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        plots_dir / "reg_shap_values.csv", index=False
    )

    for plot_type, suffix, title_suffix in [
        ("dot", "reg_shap_summary", "beeswarm — blast_radius_final"),
        ("bar", "reg_shap_bar",     "mean |SHAP| — blast_radius_final"),
    ]:
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                          plot_type=plot_type, show=False, plot_size=None)
        plt.title(f"SHAP {title_suffix} — {model_name}", fontsize=11)
        plt.tight_layout()
        path = plots_dir / f"{suffix}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved → %s", path)


# Permutation Importance

def _permutation_importance_plot(pipe, X: pd.DataFrame, y: np.ndarray,
                                 groups: np.ndarray, feature_names: list[str],
                                 plots_dir: Path, model_name: str,
                                 n_repeats: int = 10) -> None:
    logger.info("Computing permutation importance for %s…", model_name)

    # Use one GroupKFold split for speed; average importance across OOF folds
    gkf = GroupKFold(n_splits=5)
    all_imp = []

    # XGBClassifier.predict() returns float32 class indices; cast to int so
    # sklearn's f1_macro scorer does not misidentify them as continuous targets.
    _scorer = make_scorer(
        lambda y_true, y_pred: f1_score(
            y_true, y_pred.astype(int), average="macro", zero_division=0
        )
    )

    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        result = permutation_importance(
            pipe, X.iloc[test_idx], y[test_idx],
            n_repeats=n_repeats,
            scoring=_scorer,
            random_state=42,
            n_jobs=-1,
        )
        all_imp.append(result.importances_mean)

    imp_mean = np.mean(all_imp, axis=0)
    imp_std  = np.std(all_imp, axis=0)

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": np.round(imp_mean, 6),
        "std":        np.round(imp_std, 6),
    }).sort_values("importance", ascending=False)

    plots_dir.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(plots_dir / "permutation_importance.csv", index=False)
    logger.info("  Saved → %s", plots_dir / "permutation_importance.csv")

    # Plot
    top_n = min(20, len(imp_df))
    top   = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38 + 1.5))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            xerr=top["std"][::-1], align="center",
            color="#4C72B0", ecolor="#888", capsize=3)
    ax.set_xlabel("Mean decrease in macro-F1 (permutation importance)")
    ax.set_title(f"Permutation Importance — {model_name}  (top {top_n})", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    path = plots_dir / "permutation_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved → %s", path)


# Main

def main(model_path: Path, data_path: Path, stage: str, task: str = "clf") -> None:
    plots_dir = REPO_ROOT / "results" / stage / "plots"

    logger.info("Loading model from %s", model_path)
    pipe = joblib.load(model_path)
    model_name = model_path.stem

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)

    # Exclude holdout experiments
    holdout_path = REPO_ROOT / "dataset" / "processed" / "holdout_ids.txt"
    if holdout_path.exists():
        holdout_ids = set(holdout_path.read_text().strip().splitlines())
        df = df[~df["experiment_id"].isin(holdout_ids)].reset_index(drop=True)
        logger.info("Using %d training rows (%d experiments) — holdout excluded",
                    len(df), df["experiment_id"].nunique())
    else:
        logger.warning("holdout_ids.txt not found — using all data (holdout not excluded)")

    numeric_features = [c for c in BASE_NUMERIC_FEATURES + FE_NUMERIC_FEATURES if c in df.columns]
    all_features     = numeric_features + CATEGORICAL_FEATURES

    X      = df[all_features]
    groups = setting_groups(df)

    # Build target and refit to get the preprocessor in fitted state
    if task == "reg":
        y = df["blast_radius_final"].values.astype(float)
    else:
        le = LabelEncoder()
        le.classes_ = np.array(SEVERITY_ORDER)
        y = le.transform(df["severity"])

    pipe.fit(X, y)
    preprocessor = pipe.named_steps["pre"]
    X_transformed = preprocessor.transform(X)

    try:
        transformed_names = [n.split("__", 1)[-1]
                             for n in preprocessor.get_feature_names_out()]
    except Exception:
        transformed_names = all_features

    logger.info("Feature matrix: %d rows × %d features", *X_transformed.shape)

    if task == "reg":
        # Regressor SHAP
        _reg_shap_plots(pipe, X_transformed, transformed_names, plots_dir, model_name)
    else:
        # Classifier SHAP + permutation importance
        _shap_plots(pipe, X_transformed, transformed_names, plots_dir, model_name)
        _permutation_importance_plot(
            pipe, X, y, groups, all_features, plots_dir, model_name
        )

    logger.info("\nAll explainability outputs → %s/", plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP + permutation importance for a trained model")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to trained model .joblib")
    parser.add_argument("--data",  type=Path,
                        default=REPO_ROOT / "dataset" / "processed" / "training_dataset_fe.csv",
                        help="Dataset CSV used for training")
    parser.add_argument("--stage", type=str, default=None,
                        help="Results sub-directory (auto-detected from model path if omitted)")
    parser.add_argument("--task",  choices=["clf", "reg"], default="clf",
                        help="clf = classifier SHAP + permutation importance  "
                             "reg = regressor SHAP  (default: clf)")
    args = parser.parse_args()

    stage = args.stage or args.model.parent.name
    main(args.model, args.data, stage, args.task)
