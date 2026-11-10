"""
Result figures for the report and slides.

Reads from results/ and saves all plots to results/plots/.

Figures:
  01_stage_progression.png    - accuracy / macro-F1 / Medium-F1 across pipeline stages
  02_per_class_f1_heatmap.png - F1 heatmap: model x severity class, coloured by stage
  03_feature_importance.png   - top-15 features of the untuned random forest classifier on FE features
  04_accuracy_vs_time.png     - proactive OOF accuracy at T=15,30,45,60s, tuned RF vs XGBoost (clf)
  05_blast_radius_mae.png     - regression MAE across all models x stages
  06_full_model_comparison.png - grouped bars: accuracy + macro-F1 for all models x stages
  07_confusion_matrices.png   - OOF confusion matrices for tuned classifier(s)
  08_holdout_comparison.png   - CV vs holdout for the chosen classifier and regressor
  09_optuna_convergence.png   - Bayesian search trial values + best-so-far curves
  10_actual_vs_predicted.png  - true vs predicted blast radius on holdout, coloured by severity
  11_mae_vs_time.png          - proactive OOF MAE at T=15,30,45,60s, tuned RF vs XGBoost (reg)

Usage:
  python scripts/visualize.py
  python scripts/visualize.py --results results/ --out results/plots/
  python scripts/visualize.py --format pdf         # save as PDF (default: png)
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_RES  = REPO_ROOT / "results"
DEFAULT_OUT  = REPO_ROOT / "results" / "plots"

# Shared style
STAGE_ORDER  = ["baseline", "fe", "fe_tuned"]
STAGE_LABELS = {"baseline": "Baseline", "fe": "Feature\nEngineering", "fe_tuned": "Tuned (FE)"}
STAGE_COLORS = {"baseline": "#4C72B0", "fe": "#DD8452", "fe_tuned": "#55A868"}

MODEL_COLORS = {
    "RandomForest": "#4C72B0",
    "XGBoost":      "#DD8452",
    "LightGBM":     "#55A868",
}

def _apply_style():
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })

def _save(fig, out_dir: Path, name: str, fmt: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _load(res_dir: Path, filename: str) -> pd.DataFrame | None:
    path = res_dir / filename
    if not path.exists():
        logger.warning("Skipping %s — file not found: %s", filename, path)
        return None
    return pd.read_csv(path)


# Figure 1: Stage progression (best model per stage)

def plot_stage_progression(df: pd.DataFrame, out_dir: Path, fmt: str):
    """
    Line chart: accuracy, macro-F1, Medium-F1 across baseline -> FE -> tuned.
    Uses the BEST model per stage (highest macro-F1).
    """
    best = (
        df.sort_values("macro_f1", ascending=False)
          .groupby("stage", sort=False)
          .first()
          .reset_index()
    )
    # Ensure stage order
    best["stage_order"] = best["stage"].map({s: i for i, s in enumerate(STAGE_ORDER)})
    best = best.sort_values("stage_order")

    metrics   = ["accuracy", "macro_f1", "Medium_f1"]
    labels    = ["Accuracy", "Macro F1", "Medium F1"]
    colors    = ["#4C72B0", "#55A868", "#DD8452"]
    markers   = ["o", "s", "^"]
    x         = range(len(best))
    x_labels  = [STAGE_LABELS.get(s, s) for s in best["stage"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, label, color, marker in zip(metrics, labels, colors, markers):
        vals = best[metric].values
        ax.plot(x, vals, marker=marker, color=color, linewidth=2,
                markersize=8, label=label)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Across Pipeline Stages\n(best model per stage)")
    ax.legend(loc="lower right")

    # Annotate which model was best at each stage
    for xi, (_, row) in zip(x, best.iterrows()):
        ax.text(xi, 0.03, row["model"], ha="center", fontsize=8,
                color="grey", style="italic")

    _save(fig, out_dir, "01_stage_progression", fmt)


# Figure 2: Per-class F1 heatmap

def plot_f1_heatmap(df: pd.DataFrame, out_dir: Path, fmt: str):
    """
    Heatmap: rows = stage/model, columns = severity class.
    """
    df = df.copy()
    df["label"] = df.apply(
        lambda r: f"{STAGE_LABELS.get(r['stage'], r['stage'])} – {r['model']}", axis=1
    )
    # Row order: stage order, then model order within stage
    order_map = {s: i for i, s in enumerate(STAGE_ORDER)}
    df["sort_key"] = df["stage"].map(order_map).fillna(99)
    df = df.sort_values(["sort_key", "model"])

    heat_data = df.set_index("label")[["Low_f1", "Medium_f1", "High_f1"]]
    heat_data.columns = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * len(df) + 1.5)))
    sns.heatmap(
        heat_data, annot=True, fmt=".3f", cmap="YlGn",
        vmin=0.0, vmax=1.0, linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.7, "label": "F1 Score"},
    )
    ax.set_title("Per-Class F1 Score by Stage and Model")
    ax.set_xlabel("Severity Class")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, out_dir, "02_per_class_f1_heatmap", fmt)


# Figure 3: Feature importance

# Feature categories for colour coding
_FE_FEATURES = {
    "fault_severity_ratio", "service_depth", "service_downstream_count",
    "service_is_gateway", "log_target_latency_mean_ms", "log_target_memory_bytes",
    "log_fault_value_numeric", "system_error_rate", "system_latency_ratio",
}

def plot_feature_importance(df: pd.DataFrame, out_dir: Path, fmt: str, top_n: int = 15):
    """
    Horizontal bar chart of top-N features for the RandomForest classifier.
    FE-derived features shown in a different colour.
    """
    model_df = df[df["model"] == "RandomForest"].copy()
    if model_df.empty:
        # Fall back to whatever model is available
        model_df = df.groupby("feature")["importance"].mean().reset_index()
        model_name = "All models (avg)"
    else:
        model_name = "RandomForest"

    top = model_df.nlargest(top_n, "importance").sort_values("importance")
    colors = ["#DD8452" if f in _FE_FEATURES else "#4C72B0" for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.42)))
    bars = ax.barh(top["feature"], top["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name} Classifier")
    ax.set_xlim(0, top["importance"].max() * 1.15)

    for bar, val in zip(bars, top["importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Baseline feature"),
        mpatches.Patch(color="#DD8452", label="Engineered feature"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    _save(fig, out_dir, "03_feature_importance", fmt)


# Figure 4: Accuracy vs time-into-fault

def _plot_baseline_line(ax, df: pd.DataFrame, model: str, metric: str, T_vals: list, label: str,
                        linestyle: str = "--"):
    """Grey line for a no-ML baseline, if its rows are present (results from tune_model.py)."""
    bdf = df[df["model"] == model]
    if bdf.empty:
        return
    vals = [bdf.loc[bdf["time_into_fault_s"] == T, metric].values[0]
            if T in bdf["time_into_fault_s"].values else np.nan for T in T_vals]
    ax.plot(T_vals, vals, marker="o", linewidth=1.5, linestyle=linestyle, color="#7F7F7F", label=label)


def plot_accuracy_vs_time(tuned_t: pd.DataFrame | None, out_dir: Path, fmt: str,
                          source: str = "Holdout"):
    """
    Accuracy at T = 15, 30, 45, 60 s for the two tuned classifiers (random forest, XGBoost).
    source is "Holdout" (the 50 held-out runs, preferred) or "OOF" (out-of-fold on the
    training runs, if the holdout files are missing); it is shown in the title so the two
    can't be mixed up.
    tuned_t = concatenation of accuracy_vs_time_{holdout,tuned}.csv from
              results/baseline_clf_rf/ and results/baseline_clf_xgb/.
    """
    if tuned_t is None:
        logger.warning("No time-into-fault data found — skipping figure 4")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    T_vals = [15, 30, 45, 60]

    for model, color in [
        ("RandomForest", "#4C72B0"),
        ("XGBoost",      "#DD8452"),
    ]:
        mdf = tuned_t[tuned_t["model"] == model]
        if mdf.empty:
            continue
        vals = [mdf.loc[mdf["time_into_fault_s"] == T, "accuracy"]
                   .values[0] if T in mdf["time_into_fault_s"].values else np.nan
                for T in T_vals]
        ax.plot(T_vals, vals, marker="s", linewidth=2, color=color,
                label=f"Tuned – {model}")
    _plot_baseline_line(ax, tuned_t, "FullRuleBaseline", "accuracy", T_vals,
                        "Full-rule baseline (no ML)")

    ax.set_xticks(T_vals)
    ax.set_xticklabels([f"T={t}s" for t in T_vals])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel(f"{source} Accuracy")
    ax.set_xlabel("Seconds into Fault Phase (Proactive Window)")
    ax.set_title(f"Proactive Accuracy vs Time into Fault — Tuned Classifiers ({source})")
    ax.legend()
    _save(fig, out_dir, "04_accuracy_vs_time", fmt)


def plot_mae_vs_time(mae_t: pd.DataFrame | None, out_dir: Path, fmt: str,
                     source: str = "Holdout"):
    """
    MAE at T = 15, 30, 45, 60 s for the two tuned regressors (random forest, XGBoost).
    source is "Holdout" (preferred) or "OOF" (fallback), as in plot_accuracy_vs_time.
    mae_t = concatenation of mae_vs_time_{holdout,tuned}.csv from
            results/fe_reg_rf/ and results/fe_reg_xgb/.
    """
    if mae_t is None:
        logger.warning("No regression time-into-fault data found — skipping figure 11")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    T_vals = [15, 30, 45, 60]

    for model, color in [
        ("RandomForest", "#4C72B0"),
        ("XGBoost",      "#DD8452"),
    ]:
        mdf = mae_t[mae_t["model"] == model]
        if mdf.empty:
            continue
        vals = [mdf.loc[mdf["time_into_fault_s"] == T, "mae"]
                   .values[0] if T in mdf["time_into_fault_s"].values else np.nan
                for T in T_vals]
        ax.plot(T_vals, vals, marker="s", linewidth=2, color=color,
                label=f"Tuned – {model}")
    _plot_baseline_line(ax, mae_t, "EarlyBlastBaseline", "mae", T_vals,
                        '"final = early" baseline (no ML)')
    _plot_baseline_line(ax, mae_t, "NoSpreadBaseline", "mae", T_vals,
                        '"always 0" baseline (no ML)', linestyle=":")

    ax.set_xticks(T_vals)
    ax.set_xlabel("Seconds into Fault Phase (Proactive Window)")
    ax.set_xticklabels([f"T={t}s" for t in T_vals])
    ax.set_ylabel(f"{source} MAE (services)")
    ax.set_title(f"Proactive Blast-Radius MAE vs Time into Fault — Tuned Regressors ({source})")
    ax.legend()
    _save(fig, out_dir, "11_mae_vs_time", fmt)


# Figure 5: Blast radius MAE comparison

def plot_blast_radius_mae(df: pd.DataFrame, out_dir: Path, fmt: str):
    stages  = [s for s in STAGE_ORDER if s in df["stage"].values]
    models  = sorted(df["model"].unique())
    x       = np.arange(len(models))
    width   = 0.8 / len(stages)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, stage in enumerate(stages):
        sdf = df[df["stage"] == stage].set_index("model")
        vals = [sdf.loc[m, "blast_mae"] if m in sdf.index else np.nan for m in models]
        offset = (i - len(stages) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=STAGE_LABELS.get(stage, stage),
                      color=STAGE_COLORS.get(stage, "#888888"))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean Absolute Error (services)")
    ax.set_title("Blast Radius Regression — MAE by Stage and Model\n(lower is better)")
    ax.legend()
    _save(fig, out_dir, "05_blast_radius_mae", fmt)


# Figure 6: Full model comparison (accuracy + macro-F1)

def plot_full_model_comparison(df: pd.DataFrame, out_dir: Path, fmt: str):
    """
    Side-by-side grouped bars: accuracy (left) and macro-F1 (right) for all
    model x stage combinations.
    """
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    models = sorted(df["model"].unique())
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.8 / len(stages)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, metric, ylabel, title in [
        (axes[0], "accuracy",  "Accuracy",  "Overall Accuracy"),
        (axes[1], "macro_f1",  "Macro F1",  "Macro-averaged F1"),
    ]:
        for i, stage in enumerate(stages):
            sdf = df[df["stage"] == stage].set_index("model")
            vals = [sdf.loc[m, metric] if m in sdf.index else np.nan for m in models]
            offset = (i - len(stages) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                          label=STAGE_LABELS.get(stage, stage),
                          color=STAGE_COLORS.get(stage, "#888888"))
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                            f"{v:.3f}", ha="center", fontsize=7.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(loc="lower right")

    fig.suptitle("Model Comparison Across Pipeline Stages", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "06_full_model_comparison", fmt)


# Figure 7: Confusion matrices for tuned models

def plot_confusion_matrices(preds_df: pd.DataFrame, out_dir: Path, fmt: str,
                            source: str = "Holdout"):
    """
    Side-by-side confusion matrices for tuned classifier models.
    source: label shown in the figure title ("Holdout" or "OOF").
    """
    CLASSES  = ["Low", "Medium", "High"]
    _PALETTES = ["Blues", "Greens", "Oranges", "Purples"]
    models   = sorted(preds_df["model"].unique())
    titles   = [f"Tuned {m}" for m in models]
    colors   = [_PALETTES[i % len(_PALETTES)] for i in range(len(models))]

    n = max(len(models), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, model, title, cmap in zip(axes, models, titles, colors):
        mdf = preds_df[preds_df["model"] == model]
        if mdf.empty:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(mdf["true_label"], mdf["pred_label"], labels=CLASSES)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion")

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)

        # Annotate cells with count and proportion
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                count = cm[i, j]
                prop  = cm_norm[i, j]
                color = "white" if prop > 0.5 else "black"
                ax.text(j, i, f"{count}\n({prop:.0%})",
                        ha="center", va="center", fontsize=11,
                        color=color, fontweight="bold")

    fig.suptitle(f"Confusion Matrices — Tuned Models ({source} Predictions)", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "07_confusion_matrices", fmt)


# Figure 8: CV vs holdout for winning models

def plot_holdout_comparison(
    holdout_clf: pd.DataFrame | None,
    holdout_reg: pd.DataFrame | None,
    final: pd.DataFrame | None,
    out_dir: Path,
    fmt: str,
):
    """
    Two-panel figure: left = clf macro-F1 (CV vs holdout), right = reg MAE (CV vs holdout).
    Shows whether the holdout result is in line with cross-validation.
    """
    CV_COLOR      = "#4C72B0"
    HOLDOUT_COLOR = "#55A868"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Classification macro-F1
    ax = axes[0]
    if holdout_clf is not None and final is not None:
        model_name = holdout_clf["model"].iloc[0]
        tuned_rows = final[final["stage"].str.contains("tuned", na=False)]
        clf_row    = tuned_rows[tuned_rows["model"] == model_name]["macro_f1"].dropna()
        cv_f1      = float(clf_row.iloc[0]) if not clf_row.empty else None
        ho_f1      = float(holdout_clf["macro_f1"].iloc[0])

        labels = ["CV (OOF)", "Holdout"]
        vals   = [cv_f1 if cv_f1 is not None else 0, ho_f1]
        colors = [CV_COLOR, HOLDOUT_COLOR]
        bars   = ax.bar(labels, vals, color=colors, width=0.45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Classification ({model_name})\nCV vs Holdout — Macro-F1")
        ax.axhline(0.8, linestyle="--", color="grey", linewidth=0.8, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "holdout_report.csv\nnot found",
                ha="center", va="center", transform=ax.transAxes)

    # Right: Regression MAE
    ax = axes[1]
    if holdout_reg is not None and final is not None:
        model_name = holdout_reg["model"].iloc[0]
        tuned_rows = final[final["stage"].str.contains("tuned", na=False)]
        reg_row    = tuned_rows[tuned_rows["model"] == model_name]["blast_mae"].dropna()
        cv_mae     = float(reg_row.iloc[0]) if not reg_row.empty else None
        ho_mae     = float(holdout_reg["mae"].iloc[0])

        labels = ["CV (OOF)", "Holdout"]
        vals   = [cv_mae if cv_mae is not None else 0, ho_mae]
        colors = [CV_COLOR, HOLDOUT_COLOR]
        bars   = ax.bar(labels, vals, color=colors, width=0.45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("MAE (services)")
        ax.set_title(f"Regression ({model_name})\nCV vs Holdout — MAE")
        if "r2" in holdout_reg.columns:
            r2 = float(holdout_reg["r2"].iloc[0])
            ax.text(0.97, 0.95, f"Holdout R² = {r2:.3f}",
                    transform=ax.transAxes, ha="right", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6))
    else:
        ax.text(0.5, 0.5, "holdout_regression_report.csv\nnot found",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Winning Models — Cross-Validation vs Independent Holdout Test Set",
                 fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "08_holdout_comparison", fmt)


# Figure 9: Optuna convergence curves

_MODEL_SHORT = {"rf": "RandomForest", "xgb": "XGBoost", "lgbm": "LightGBM"}

def plot_optuna_convergence(res_dir: Path, out_dir: Path, fmt: str):
    """
    Two-panel convergence plot built from trial_history_*.csv files saved by tune_model.py.
    Auto-discovers files across all stage subdirectories under res_dir.
    Left: classification (macro-F1, maximize); Right: regression (MAE, minimize).
    """
    clf_histories: dict[str, pd.DataFrame] = {}
    reg_histories: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(res_dir.glob("*/trial_history_*.csv")):
        # filename pattern: trial_history_{model}_{task}.csv
        parts = csv_path.stem.split("_")  # ["trial","history","xgb","clf"]
        if len(parts) < 4:
            continue
        task  = parts[-1]
        model = _MODEL_SHORT.get(parts[-2], parts[-2])
        df    = pd.read_csv(csv_path)
        if task == "clf":
            clf_histories[model] = df
        elif task == "reg":
            reg_histories[model] = df

    if not clf_histories and not reg_histories:
        logger.warning("No trial_history_*.csv files found — skipping figure 09")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, histories, ylabel, direction_label in [
        (axes[0], clf_histories, "Macro-F1",       "classification — maximize"),
        (axes[1], reg_histories, "MAE (services)",  "regression — minimize"),
    ]:
        if not histories:
            ax.text(0.5, 0.5, "No trial history found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Optuna Convergence\n({direction_label})")
            continue

        for model, df in sorted(histories.items()):
            color = MODEL_COLORS.get(model, "#888888")
            ax.scatter(df["number"], df["value"],
                       alpha=0.25, s=14, color=color)
            ax.plot(df["number"], df["best_so_far"],
                    linewidth=2, color=color, label=model)

        ax.set_xlabel("Trial Number")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Optuna Convergence\n({direction_label})")
        ax.legend()

    fig.suptitle("Bayesian Hyperparameter Search — 50 Trials per Model", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "09_optuna_convergence", fmt)


# Figure 10: Actual vs Predicted scatter (holdout regression)

def plot_actual_vs_predicted(
    reg_preds: pd.DataFrame,
    holdout_reg: pd.DataFrame | None,
    out_dir: Path,
    fmt: str,
):
    """
    Scatter plot of true vs predicted blast_radius_final on the holdout set.
    Points are coloured by severity class and jittered horizontally to avoid
    overplotting (true values are integers).
    """
    SEVERITY_COLORS = {"Low": "#55A868", "Medium": "#DD8452", "High": "#C00000"}

    true_vals  = reg_preds["true_blast_radius"].values.astype(float)
    pred_vals  = reg_preds["pred_blast_radius"].values.astype(float)
    severities = reg_preds["severity"].values if "severity" in reg_preds.columns else None

    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(true_vals))

    fig, ax = plt.subplots(figsize=(7, 6))

    if severities is not None:
        for sev, color in SEVERITY_COLORS.items():
            mask = severities == sev
            if mask.any():
                ax.scatter(
                    true_vals[mask] + jitter[mask], pred_vals[mask],
                    color=color, label=sev, alpha=0.75, s=65,
                    edgecolors="white", linewidth=0.5,
                )
        ax.legend(title="Severity", frameon=True, fontsize=10)
    else:
        ax.scatter(true_vals + jitter, pred_vals, color="#4C72B0", alpha=0.75, s=65)

    lo = min(true_vals.min(), pred_vals.min()) - 0.5
    hi = max(true_vals.max(), pred_vals.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="Perfect prediction")

    if holdout_reg is not None:
        mae  = float(holdout_reg["mae"].iloc[0])
        rmse = float(holdout_reg["rmse"].iloc[0])
        r2   = float(holdout_reg["r2"].iloc[0]) if "r2" in holdout_reg.columns else None
        txt  = f"MAE  = {mae:.3f}\nRMSE = {rmse:.3f}"
        if r2 is not None:
            txt += f"\nR²   = {r2:.3f}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("True Blast Radius (services degraded)", fontsize=12)
    ax.set_ylabel("Predicted Blast Radius", fontsize=12)
    ax.set_title(
        "Blast Radius — Actual vs Predicted  ·  Holdout Set (20 experiments × 4 snapshots)",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "10_actual_vs_predicted", fmt)


# Main

def main(res_dir: Path, out_dir: Path, fmt: str, stage: str = "fe_tuned"):
    _apply_style()

    # tuned_dir: results/<stage>/  - final_results.csv, oof, accuracy_vs_time_tuned
    # fe_dir:    results/<fe>/     - feature_importance, accuracy_vs_time
    tuned_dir = res_dir / stage
    fe_dir    = res_dir / stage.replace("_tuned", "")
    logger.info("Reading results from %s", tuned_dir)

    # Load available data files from the correct stage subdirectories
    final       = _load(tuned_dir, "final_results.csv")
    feat_imp    = _load(fe_dir,    "feature_importance.csv")
    oof_preds      = _load(tuned_dir, "oof_predictions.csv")
    holdout_preds  = _load(tuned_dir, "holdout_predictions.csv")
    holdout_clf    = _load(tuned_dir, "holdout_report.csv")
    holdout_reg    = _load(tuned_dir, "holdout_regression_report.csv")
    reg_preds      = _load(tuned_dir, "holdout_regression_predictions.csv")

    # accuracy/MAE per T is in the tuning folders of each model, not in the gathered stage
    # folder. The holdout numbers are used when present, otherwise the out-of-fold ones.
    def _load_concat(dirs_and_file):
        dfs = [_load(d, f) for d, f in dirs_and_file]
        dfs = [df for df in dfs if df is not None]
        return pd.concat(dfs, ignore_index=True) if dfs else None

    acc_t_source = "Holdout"
    acc_t_tu = _load_concat([
        (res_dir / "baseline_clf_rf",  "accuracy_vs_time_holdout.csv"),
        (res_dir / "baseline_clf_xgb", "accuracy_vs_time_holdout.csv"),
    ])
    if acc_t_tu is None:
        acc_t_source = "OOF"
        acc_t_tu = _load_concat([
            (res_dir / "baseline_clf_rf",  "accuracy_vs_time_tuned.csv"),
            (res_dir / "baseline_clf_xgb", "accuracy_vs_time_tuned.csv"),
        ])

    mae_t_source = "Holdout"
    mae_t_tu = _load_concat([
        (res_dir / "fe_reg_rf",  "mae_vs_time_holdout.csv"),
        (res_dir / "fe_reg_xgb", "mae_vs_time_holdout.csv"),
    ])
    if mae_t_tu is None:
        mae_t_source = "OOF"
        mae_t_tu = _load_concat([
            (res_dir / "fe_reg_rf",  "mae_vs_time_tuned.csv"),
            (res_dir / "fe_reg_xgb", "mae_vs_time_tuned.csv"),
        ])

    # Figure 1: Stage progression
    if final is not None:
        plot_stage_progression(final, out_dir, fmt)
    else:
        logger.warning("final_results.csv missing — run: python scripts/tune_model.py")

    # Figure 2: Per-class F1 heatmap
    if final is not None:
        plot_f1_heatmap(final, out_dir, fmt)

    # Figure 3: Feature importance
    if feat_imp is not None:
        plot_feature_importance(feat_imp, out_dir, fmt)

    # Figure 4: Accuracy vs time-into-fault (tuned classifiers)
    plot_accuracy_vs_time(acc_t_tu, out_dir, fmt, source=acc_t_source)

    # Figure 5: Blast radius MAE
    if final is not None:
        plot_blast_radius_mae(final, out_dir, fmt)

    # Figure 6: Full model comparison
    if final is not None:
        plot_full_model_comparison(final, out_dir, fmt)

    # Figure 7: Confusion matrices (holdout preferred over OOF)
    if holdout_preds is not None:
        plot_confusion_matrices(holdout_preds, out_dir, fmt, source="Holdout")
    elif oof_preds is not None:
        logger.warning("holdout_predictions.csv missing — falling back to OOF predictions")
        plot_confusion_matrices(oof_preds, out_dir, fmt, source="OOF")
    else:
        logger.warning("holdout_predictions.csv missing — run: python scripts/tune_model.py --task clf --eval-only")

    # Figure 8: CV vs holdout comparison
    if holdout_clf is not None or holdout_reg is not None:
        plot_holdout_comparison(holdout_clf, holdout_reg, final, out_dir, fmt)
    else:
        logger.warning("holdout_report.csv / holdout_regression_report.csv missing")

    # Figure 9: Optuna convergence
    plot_optuna_convergence(res_dir, out_dir, fmt)

    # Figure 10: Actual vs Predicted scatter (holdout regression)
    if reg_preds is not None:
        plot_actual_vs_predicted(reg_preds, holdout_reg, out_dir, fmt)
    else:
        logger.warning("holdout_regression_predictions.csv missing — run: "
                       "python scripts/tune_model.py --task reg --eval-only")

    # Figure 11: MAE vs time-into-fault (tuned regressors)
    plot_mae_vs_time(mae_t_tu, out_dir, fmt, source=mae_t_source)

    logger.info("\nAll figures saved to %s/", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dissertation visualizations")
    parser.add_argument("--results", type=Path, default=DEFAULT_RES,
                        help="Root results/ directory (default: results/)")
    parser.add_argument("--stage",   type=str,  default="fe_tuned",
                        help="Tuned-stage sub-directory under results/ "
                             "(default: fe_tuned). FE files are loaded from "
                             "results/<stage without _tuned>/.")
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT,
                        help="Output directory for plots")
    parser.add_argument("--format",  choices=["png", "pdf", "svg"], default="png",
                        help="Output file format (default: png)")
    args = parser.parse_args()
    main(args.results, args.out, args.format, stage=args.stage)
