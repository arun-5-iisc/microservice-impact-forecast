"""
Generate dissertation-quality visualizations for the three-stage ML pipeline.

Reads from results/ and saves all plots to results/plots/.

Figures:
  01_stage_progression.png   — accuracy / macro-F1 / Medium-F1 across pipeline stages
  02_per_class_f1_heatmap.png — F1 heatmap: model × severity class, coloured by stage
  03_feature_importance.png  — top-15 features from best classifier (RF on FE)
  04_accuracy_vs_time.png    — proactive accuracy at T=15,30,45,60s across stages
  05_blast_radius_mae.png    — regression MAE across all models × stages
  06_full_model_comparison.png — grouped bars: accuracy + macro-F1 for all models × stages

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_RES  = REPO_ROOT / "results"
DEFAULT_OUT  = REPO_ROOT / "results" / "plots"

# ── Shared style ──────────────────────────────────────────────────────────────
STAGE_ORDER  = ["baseline", "fe", "tuned_fe"]
STAGE_LABELS = {"baseline": "Baseline", "fe": "Feature\nEngineering", "tuned_fe": "Tuned (FE)"}
STAGE_COLORS = {"baseline": "#4C72B0", "fe": "#DD8452", "tuned_fe": "#55A868"}

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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Stage progression (best model per stage)
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage_progression(df: pd.DataFrame, out_dir: Path, fmt: str):
    """
    Line chart: accuracy, macro-F1, Medium-F1 across baseline → FE → tuned.
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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-class F1 heatmap
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Feature importance
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Accuracy vs time-into-fault
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_time(baseline_t: pd.DataFrame | None, tuned_t: pd.DataFrame | None,
                          out_dir: Path, fmt: str):
    """
    Line chart: proactive accuracy at T=15,30,45,60s.
    baseline_t = accuracy_vs_time.csv (LightGBM OOF from train_model.py)
    tuned_t    = accuracy_vs_time_tuned.csv (RF + LightGBM tuned from tune_model.py)
    """
    if baseline_t is None and tuned_t is None:
        logger.warning("No time-into-fault data found — skipping figure 4")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    T_vals = [15, 30, 45, 60]

    if baseline_t is not None:
        vals = [baseline_t.loc[baseline_t["time_into_fault_s"] == T, "accuracy"]
                          .values[0] if T in baseline_t["time_into_fault_s"].values else np.nan
                for T in T_vals]
        ax.plot(T_vals, vals, marker="o", linewidth=2, color="#4C72B0",
                label="Baseline – LightGBM", linestyle="--")

    if tuned_t is not None:
        for model, color, ls in [
            ("RandomForest", "#DD8452", "-"),
            ("LightGBM",     "#55A868", "-"),
        ]:
            mdf = tuned_t[tuned_t["model"] == model]
            if mdf.empty:
                continue
            vals = [mdf.loc[mdf["time_into_fault_s"] == T, "accuracy"]
                       .values[0] if T in mdf["time_into_fault_s"].values else np.nan
                    for T in T_vals]
            ax.plot(T_vals, vals, marker="s", linewidth=2, color=color,
                    label=f"Tuned – {model}", linestyle=ls)

    ax.set_xticks(T_vals)
    ax.set_xticklabels([f"T={t}s" for t in T_vals])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("OOF Accuracy")
    ax.set_xlabel("Seconds into Fault Phase (Proactive Window)")
    ax.set_title("Proactive Accuracy vs Time into Fault")
    ax.legend()
    ax.axhline(0.758, color="#4C72B0", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Baseline overall acc.")
    _save(fig, out_dir, "04_accuracy_vs_time", fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Blast radius MAE comparison
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Full model comparison (accuracy + macro-F1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_model_comparison(df: pd.DataFrame, out_dir: Path, fmt: str):
    """
    Side-by-side grouped bars: accuracy (left) and macro-F1 (right) for all
    model × stage combinations.
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(res_dir: Path, out_dir: Path, fmt: str):
    _apply_style()

    logger.info("Reading results from %s", res_dir)

    # Load available data files
    final    = _load(res_dir, "final_results.csv")
    feat_imp = _load(res_dir, "feature_importance.csv")
    acc_t    = _load(res_dir, "accuracy_vs_time.csv")
    acc_t_tu = _load(res_dir, "accuracy_vs_time_tuned.csv")

    # ── Figure 1: Stage progression ──────────────────────────────────────────
    if final is not None:
        plot_stage_progression(final, out_dir, fmt)
    else:
        logger.warning("final_results.csv missing — run: python scripts/tune_model.py --eval-only")

    # ── Figure 2: Per-class F1 heatmap ───────────────────────────────────────
    if final is not None:
        plot_f1_heatmap(final, out_dir, fmt)

    # ── Figure 3: Feature importance ─────────────────────────────────────────
    if feat_imp is not None:
        plot_feature_importance(feat_imp, out_dir, fmt)

    # ── Figure 4: Accuracy vs time-into-fault ────────────────────────────────
    if acc_t is not None or acc_t_tu is not None:
        plot_accuracy_vs_time(acc_t, acc_t_tu, out_dir, fmt)

    # ── Figure 5: Blast radius MAE ───────────────────────────────────────────
    if final is not None:
        plot_blast_radius_mae(final, out_dir, fmt)

    # ── Figure 6: Full model comparison ──────────────────────────────────────
    if final is not None:
        plot_full_model_comparison(final, out_dir, fmt)

    logger.info("\nAll figures saved to %s/", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dissertation visualizations")
    parser.add_argument("--results", type=Path, default=DEFAULT_RES,
                        help="Directory containing results CSVs")
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT,
                        help="Output directory for plots")
    parser.add_argument("--format",  choices=["png", "pdf", "svg"], default="png",
                        help="Output file format (default: png)")
    args = parser.parse_args()
    main(args.results, args.out, args.format)
